"""Ampere grouped-query attention for one long bidirectional span."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["nq", "nk"])
def _grouped(Q, K, V, Y, nq, nk, scale: tl.constexpr, Q0: tl.constexpr, Q1: tl.constexpr,
             K0: tl.constexpr, K1: tl.constexpr, V0: tl.constexpr, V1: tl.constexpr,
             H: tl.constexpr, G: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    flattened = tl.program_id(0) * BM + tl.arange(0, BM)
    kv_head = tl.program_id(1)
    row, head = flattened // G, kv_head * G + flattened % G
    k_col = tl.arange(0, BN)
    dim = tl.arange(0, D)
    q = tl.load(Q + row[:, None] * Q0 + head[:, None] * Q1 + dim[None, :], row[:, None] < nq, 0)
    m = tl.full((BM,), float("-inf"), tl.float32)
    denom = tl.full((BM,), 0, tl.float32)
    out = tl.full((BM, D), 0, tl.float32)
    for first in range(0, nk, BN):
        key_rows = first + k_col
        k = tl.load(K + key_rows[None, :] * K0 + kv_head * K1 + dim[:, None], key_rows[None, :] < nk, 0)
        z = tl.dot(q, k).to(tl.float32) * (scale * 1.4426950408889634)
        z = tl.where(key_rows[None, :] < nk, z, float("-inf"))
        new_m = tl.maximum(m, tl.max(z, 1))
        p = tl.exp2(z - new_m[:, None])
        alpha = tl.exp2(m - new_m)
        denom = denom * alpha + tl.sum(p, 1)
        v = tl.load(V + key_rows[:, None] * V0 + kv_head * V1 + dim[None, :], key_rows[:, None] < nk, 0)
        out = tl.dot(p.to(v.dtype), v, out * alpha[:, None])
        m = new_m
    y = out / denom[:, None]
    tl.store(Y + (row[:, None] * H + head[:, None]) * D + dim[None, :], y, row[:, None] < nq)


def grouped(query, key, value, scale, tile=(32, 32, 8, 2)):
    rows, heads, dim = query.shape
    kv_heads = key.shape[1]
    group = heads // kv_heads
    bm, bn, warps, stages = tile
    out = torch.empty((rows, heads, dim), dtype=query.dtype, device=query.device)
    grid = triton.cdiv(rows * group, bm), kv_heads
    _grouped[grid](query, key, value, out, rows, key.shape[0], scale,
                   *query.stride()[:2], *key.stride()[:2], *value.stride()[:2],
                   heads, group, dim, bm, bn, num_warps=warps, num_stages=stages)
    return out
