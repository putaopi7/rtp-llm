# SPDX-License-Identifier: MIT
"""Unit test: chunked vs non-chunked sparse attention precision.

Validates that ``_sparse_attn_chunked`` (the M3_SPARSE_ATTN_CHUNK_ENABLE
path in ``topk_bt_fused.py``) produces attention output matching the full
non-chunked ``sparse_atten_func`` reference. Chunking over the query dim is
lossless by construction (each query row only depends on its own top-k KV
blocks), so the two must agree to within bf16 numerical noise.

Covered cases:
  * single request, total_q divisible by chunk_size (even partition),
  * single request, total_q NOT divisible by chunk_size (trailing partial
    chunk -- exercises the boundary where the last chunk is smaller),
  * multi-batch ragged request (B=2, different q/k lengths) -- exercises
    per-batch cu_seqlens / page_table / seqused_k partitioning.
"""
import unittest

import torch


def _sm100_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import fmha_sm100  # noqa: F401
    except Exception:
        return False
    try:
        return torch.cuda.get_device_capability(0)[0] == 10
    except Exception:
        return False


def _make_topk_idx(
    nkv: int, total_q: int, topk: int, blk: int, device, seed: int
) -> torch.Tensor:
    """Causal-valid random topk block ids with -1 padding.

    Block id visible at query i is < i//blk + 1 (causal); slots beyond the
    per-query visible count are padded with -1.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    num_blocks = (total_q + blk - 1) // blk
    avail = torch.arange(total_q) // blk + 1  # blocks visible at query i
    scores = torch.rand(nkv, total_q, num_blocks, generator=g)
    scores = scores.masked_fill(
        torch.arange(num_blocks)[None, None, :] >= avail[None, :, None], -1.0
    )
    idx = scores.topk(min(topk, num_blocks), dim=-1).indices.int()
    if idx.shape[-1] < topk:
        idx = torch.cat(
            [idx, idx.new_full((nkv, total_q, topk - idx.shape[-1]), -1)], dim=-1
        )
    pad = torch.arange(topk)[None, None, :] >= avail[None, :, None]
    return idx.masked_fill(pad, -1).contiguous().to(device)


def _make_topk_idx_varlen(
    nkv: int, qo_lens, topk: int, blk: int, device, seed: int
) -> torch.Tensor:
    """Per-batch causal topk_idx concatenated along dim 1 ([nkv, total_q, topk])."""
    parts = [
        _make_topk_idx(nkv, ql, topk, blk, device, seed + b)
        for b, ql in enumerate(qo_lens)
    ]
    return torch.cat(parts, dim=1)


def _aligned_page_table(ids_per_batch, device) -> torch.Tensor:
    """Build a [B, max_n] int32 page table whose last dim is contiguous and
    whose every row is 16-byte aligned (the cute kernel asserts %16 == 0 on
    each row's base). Rows are padded with -1 beyond each batch's page count.
    """
    B = len(ids_per_batch)
    max_n = max(len(x) for x in ids_per_batch)
    # Pad the row width up to a multiple of 4 int32 (=16 bytes) so row 1..B-1
    # stay 16-byte aligned when laid out contiguously.
    max_n_pad = ((max_n + 3) // 4) * 4
    pt = torch.full((B, max_n_pad), -1, dtype=torch.int32, device=device)
    for b, ids in enumerate(ids_per_batch):
        pt[b, : len(ids)] = ids
    return pt[:, :max_n]


def _run_full_reference(
    builder,
    q,
    k_paged,
    v_paged,
    topk_idx,
    qo_lens,
    kv_lens,
    blk,
    topk,
    sm_scale,
    device,
):
    """Non-chunked reference: one CSR build + one sparse_atten_func call over
    the full (possibly multi-batch) sequence."""
    from interface import sparse_atten_func

    nkv = k_paged.shape[1]
    hq = q.shape[1]
    hkv = nkv
    total_q = q.shape[0]
    total_k = int(sum(kv_lens))
    cu_q = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(qo_lens), 0).tolist()),
        dtype=torch.int32,
        device=device,
    )
    cu_k = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(kv_lens), 0).tolist()),
        dtype=torch.int32,
        device=device,
    )
    total_rows = (total_k + blk - 1) // blk
    sk = torch.tensor(kv_lens, dtype=torch.int32, device=device)

    # Physical page ids per batch: contiguous [0..n0-1], [n0..n0+n1-1], ...
    off = 0
    ids_per_batch = []
    for kl in kv_lens:
        n = (kl + blk - 1) // blk
        ids_per_batch.append(
            torch.arange(off, off + n, dtype=torch.int32, device=device)
        )
        off += n
    pt = _aligned_page_table(ids_per_batch, device)

    row_ptr, q_ind, sched = builder(
        topk_idx,
        cu_q,
        cu_k,
        total_k=total_k,
        blk_kv=blk,
        max_seqlen_k=total_k,
        max_seqlen_q=max(qo_lens),
        total_rows=total_rows,
        qhead_per_kv=hq // hkv,
        return_schedule=True,
    )
    out = sparse_atten_func(
        q,
        k_paged,
        v_paged,
        row_ptr,
        q_ind,
        topk,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(qo_lens),
        max_seqlen_k=total_k,
        blk_kv=blk,
        causal=True,
        softmax_scale=sm_scale,
        return_softmax_lse=False,
        page_table=pt,
        seqused_k=sk,
        schedule=sched,
        usable_SM_count=-1,
    )
    return out[0] if isinstance(out, tuple) else out


@unittest.skipUnless(_sm100_available(), "SM100 CUDA device + fmha_sm100 required")
class SparsePrefillChunkTest(unittest.TestCase):
    HQ, HKV, DIM, BLK, TOPK = 64, 4, 128, 128, 16

    def setUp(self):
        # Importing fmha_sm100 eagerly sets up the cute/ sys.path entries that
        # ``_sparse_attn_chunked`` relies on for its lazy ``from src.sm100...``
        # / ``from interface import ...`` imports.
        import fmha_sm100  # noqa: F401

        from rtp_llm.models_py.triton_kernels.sparse_msa.prefill import (
            topk_bt_fused as tbf,
        )

        self.tbf = tbf
        # Reset the per-device workspace cache so each case starts clean and
        # so we can assert reuse within a case.
        tbf._M3_CHUNK_WS_CACHE.clear()
        torch.cuda.set_device(0)

    def _make_inputs(self, qo_lens, device, seed=0, kv_dtype=torch.bfloat16):
        from src.sm100.prepare_k2q_csr import SparseK2qCsrBuilderSm100

        nkv, hq, dim, blk, topk = self.HKV, self.HQ, self.DIM, self.BLK, self.TOPK
        total_q = int(sum(qo_lens))
        total_k = total_q  # causal, prefix=0
        num_pages = (total_k + blk - 1) // blk
        torch.manual_seed(seed)
        q = torch.randn(total_q, hq, dim, dtype=torch.bfloat16, device=device)
        # fp8_e4m3 range is ~[-448, 448] but precision collapses outside ~[-2, 2];
        # scale randn down so the fp8 KV cache holds realistic (non-saturated)
        # values. Q stays bf16 -- the kernel's mixed fp8-KV / bf16-Q path.
        kv_scale = 0.3 if kv_dtype == torch.float8_e4m3fn else 1.0
        k = (torch.randn(num_pages, nkv, blk, dim, device=device) * kv_scale).to(
            kv_dtype
        )
        v = (torch.randn(num_pages, nkv, blk, dim, device=device) * kv_scale).to(
            kv_dtype
        )
        sm_scale = dim**-0.5
        topk_idx = _make_topk_idx_varlen(
            nkv, list(qo_lens), topk, blk, device, seed + 1
        )
        kv_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
        builder = SparseK2qCsrBuilderSm100()
        builder._ensure_loaded()
        return q, k, v, topk_idx, kv_indices, builder, sm_scale

    def _run_chunked(
        self, q, k, v, topk_idx, kv_indices, qo_lens, chunk_size, sm_scale, device
    ):
        nkv = self.HKV
        plan = dict(
            num_kv_heads=nkv,
            qo_segment_lens=torch.tensor(list(qo_lens), dtype=torch.int32),
            seqused_k=torch.tensor(list(qo_lens), dtype=torch.int32, device=device),
            causal=True,
            usable_SM_count=-1,
        )
        out = self.tbf._sparse_attn_chunked(
            q,
            k,
            v,
            topk_idx,
            kv_indices,
            plan,
            self.TOPK,
            self.BLK,
            sm_scale,
            chunk_size,
        )
        # Drop the per-forward chunk metadata so the next case rebuilds it.
        plan.pop("_chunk_meta", None)
        return out

    def _assert_close(self, out, ref, case_name):
        both = torch.cat([out.reshape(-1), ref.reshape(-1)])
        self.assertFalse(
            torch.isnan(both).any() or torch.isinf(both).any(),
            f"{case_name}: non-finite output",
        )
        max_abs = (out.float() - ref.float()).abs().max().item()
        cos = torch.nn.functional.cosine_similarity(
            out.reshape(1, -1).float(), ref.reshape(1, -1).float(), dim=1
        ).item()
        bitwise = torch.equal(out, ref)
        print(f"[{case_name}] max_abs={max_abs:.3e} cos={cos:.6f} bitwise={bitwise}")
        self.assertGreater(
            cos, 0.999, f"{case_name}: cosine similarity {cos:.6f} below 0.999"
        )
        # bf16 attention with a different reduction schedule can differ in the
        # last bit; 0.05 is a comfortable bound well above noise and well below
        # any real regression (a chunking bug would diverge by O(1)).
        self.assertLess(max_abs, 0.05, f"{case_name}: max_abs {max_abs:.3e} too large")

    def test_even_partition(self):
        device = torch.device("cuda", 0)
        qo_lens = [8192]
        q, k, v, topk_idx, kv_indices, builder, sm = self._make_inputs(qo_lens, device)
        ref = _run_full_reference(
            builder,
            q,
            k,
            v,
            topk_idx,
            qo_lens,
            qo_lens,
            self.BLK,
            self.TOPK,
            sm,
            device,
        )
        out = self._run_chunked(
            q, k, v, topk_idx, kv_indices, qo_lens, 2048, sm, device
        )
        n_chunks = len(self.tbf._M3_CHUNK_WS_CACHE)  # cache populated
        self._assert_close(out, ref, "even_partition")
        self.assertGreater(n_chunks, 0)

    def test_uneven_trailing_chunk(self):
        device = torch.device("cuda", 0)
        # 5120 = 4096 + 1024: last chunk is a quarter of chunk_size.
        qo_lens = [5120]
        q, k, v, topk_idx, kv_indices, builder, sm = self._make_inputs(
            qo_lens, device, seed=7
        )
        ref = _run_full_reference(
            builder,
            q,
            k,
            v,
            topk_idx,
            qo_lens,
            qo_lens,
            self.BLK,
            self.TOPK,
            sm,
            device,
        )
        out = self._run_chunked(
            q, k, v, topk_idx, kv_indices, qo_lens, 4096, sm, device
        )
        self._assert_close(out, ref, "uneven_trailing_chunk")

    def test_multi_batch_ragged(self):
        device = torch.device("cuda", 0)
        qo_lens = [3072, 2048]
        q, k, v, topk_idx, kv_indices, builder, sm = self._make_inputs(
            qo_lens, device, seed=13
        )
        ref = _run_full_reference(
            builder,
            q,
            k,
            v,
            topk_idx,
            qo_lens,
            qo_lens,
            self.BLK,
            self.TOPK,
            sm,
            device,
        )
        out = self._run_chunked(
            q, k, v, topk_idx, kv_indices, qo_lens, 2048, sm, device
        )
        self._assert_close(out, ref, "multi_batch_ragged")

    def test_fp8_kv_cache(self):
        # Mixed fp8_e4m3fn KV cache + bf16 Q: the kernel auto-detects this via
        # _resolve_forward_mma_dtypes (qk_dtype=bf16, pv_dtype=bf16). Verifies
        # chunking is lossless on the fp8-KV path too -- both the chunked call
        # and the full reference consume the SAME fp8 K/V, so they must agree.
        device = torch.device("cuda", 0)
        qo_lens = [8192]
        q, k, v, topk_idx, kv_indices, builder, sm = self._make_inputs(
            qo_lens, device, seed=21, kv_dtype=torch.float8_e4m3fn
        )
        self.assertEqual(k.dtype, torch.float8_e4m3fn)
        self.assertEqual(v.dtype, torch.float8_e4m3fn)
        ref = _run_full_reference(
            builder,
            q,
            k,
            v,
            topk_idx,
            qo_lens,
            qo_lens,
            self.BLK,
            self.TOPK,
            sm,
            device,
        )
        out = self._run_chunked(
            q, k, v, topk_idx, kv_indices, qo_lens, 2048, sm, device
        )
        self._assert_close(out, ref, "fp8_kv_cache")


if __name__ == "__main__":
    unittest.main()
