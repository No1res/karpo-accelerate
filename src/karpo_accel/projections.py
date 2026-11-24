"""Fused W8A8 attention projections and rotated W4A4 feed-forward layers."""

import torch
import triton
import triton.language as tl
from karpo_decision.families.gemma import (
    fused,
    global_attention,
    sliding_attention,
    vision,
)
from karpo_decision.families.gemma.packed import PaddedPlan
from torch import nn
from triton.language.extra.cuda import libdevice


@triton.jit
def _quantize8(X, Q, S, width: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, block)
    offset = row * width + cols
    values = tl.load(X + offset, cols < width, 0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(values), 0) / 127.0, 1.0e-12)
    tl.store(Q + offset, libdevice.rint(values / scale).to(tl.int8), cols < width)
    tl.store(S + row, scale)


@triton.jit
def _rotate_and_pack4(X, Signs, Q, S, width: tl.constexpr, groups: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, 32)[None, :]
    index = tl.arange(0, groups)[:, None] * 32 + lane
    valid = index < width
    values = tl.load(X + row * width + index, valid, 0).to(tl.float32)
    values *= tl.load(Signs + index, valid, 1)
    for level in tl.static_range(5):
        step = 1 << level
        partner = tl.gather(values, tl.broadcast_to(lane ^ step, (groups, 32)), 1)
        values = tl.where((lane & step) == 0, values + partner, partner - values)
    scale = tl.maximum(tl.max(tl.max(tl.abs(values), 1), 0) / 7, 1.0e-12)
    low = libdevice.rint(values / scale).to(tl.int32) & 15
    high = tl.gather(low, tl.broadcast_to(lane ^ 1, (groups, 32)), 1)
    tl.store(Q + row * (width // 2) + index // 2, low | (high << 4), valid & ((lane & 1) == 0))
    tl.store(S + row, scale * 0.1767766952966369)


def quantize8(value):
    matrix = value.reshape(-1, value.shape[-1]).contiguous()
    rows, width = matrix.shape
    values = torch.empty_like(matrix, dtype=torch.int8)
    scales = torch.empty(rows, dtype=torch.float32, device=matrix.device)
    _quantize8[(rows,)](matrix, values, scales, width, triton.next_power_of_2(width),
                        num_warps=4 if width <= 8192 else 8)
    return values, scales


def quantize_rotated4(value, signs):
    matrix = value.reshape(-1, value.shape[-1]).contiguous()
    rows, width = matrix.shape
    if width % 32 or signs.numel() != width:
        raise ValueError("Hadamard blocks need one sign per input and a width divisible by 32")
    packed = torch.empty((rows, width // 2), device=matrix.device, dtype=torch.uint8)
    scales = torch.empty(rows, device=matrix.device, dtype=torch.float32)
    _rotate_and_pack4[(rows,)](matrix, signs, packed, scales, width, triton.next_power_of_2(width // 32),
                              num_warps=8)
    return packed, scales


class Int8Projection(nn.Module):
    def __init__(self, projections):
        super().__init__()
        width = projections[0].weight.shape[1]
        if any(module.bias is not None or module.weight.shape[1] != width for module in projections):
            raise ValueError("Projection fusion needs a shared input width and bias-free weights")
        weight, scales = zip(*(quantize8(module.weight) for module in projections), strict=True)
        self.register_buffer("weight", torch.cat(weight))
        self.register_buffer("scales", torch.cat(scales))

    def forward(self, value):
        q, scales = quantize8(value)
        channels = self.weight.shape[0]
        result = torch.empty((q.shape[0], channels), device=q.device, dtype=torch.bfloat16)
        torch.ops.karpo_ampere.scaled_mm(result, q, self.weight.t(), scales.view(-1, 1), self.scales.view(1, -1))
        return result.view(*value.shape[:-1], channels)


def apply_rotation(width, device):
    """Return the fixed signs used for both inputs and weights in this profile."""
    rng = torch.Generator(device=device).manual_seed(1913 + width)
    return torch.randint(0, 2, (width,), device=device, generator=rng, dtype=torch.int8).mul_(2).sub_(1)


class Int4Projection(nn.Module):
    def __init__(self, projection):
        super().__init__()
        weight = projection.weight.float().mul_(projection.scales[:, None])
        signs = apply_rotation(weight.shape[1], weight.device)
        packed, scales = quantize_rotated4(weight, signs)
        self.register_buffer("signs", signs)
        self.register_buffer("packed", packed)
        self.register_buffer("scales", scales)

    def forward(self, value):
        q, scale = quantize_rotated4(value, self.signs)
        rows, channels = q.shape[0], self.packed.shape[0]
        result = torch.empty((rows, channels), device=q.device, dtype=torch.bfloat16)
        tile = 64 if rows <= 320 and channels < 8192 else 128
        torch.ops.karpo_ampere_int4_fused.scaled_mm(result, q, self.packed, scale, self.scales, tile)
        return result.view(*value.shape[:-1], channels)


def project_attention(attn, x, cos, sin):
    qkv = attn.qkv_proj(x).split(attn.qkv_widths, dim=-1)
    shape = 1, x.shape[1], -1, attn.head_dim
    raw_q, raw_k = qkv[0].reshape(shape), qkv[1].reshape(shape)
    raw_v = qkv[2].reshape(shape) if len(qkv) == 3 else raw_k
    query = fused.norm_rope(attn.q_norm, raw_q, cos, sin)
    key = fused.norm_rope(attn.k_norm, raw_k, cos, sin)
    value = fused.norm(attn.v_norm, raw_v)
    return query, key, value


@torch.no_grad()
def install_projections(model, profile):
    """Replace body projections. Keep the marker-only final layer at its original precision."""
    if profile not in ("w8a8", "w4a4"):
        raise ValueError(f"Unknown quantized profile: {profile}")

    for layer in model.layers[:-1]:
        mlp = layer.mlp
        gate_up = Int8Projection((mlp.gate_proj, mlp.up_proj))
        down = Int8Projection((mlp.down_proj,))
        if profile == "w4a4":
            # The evaluated W4A4 recipe starts from the same row-quantized weights as W8A8.
            gate_up = Int4Projection(gate_up)
            down = Int4Projection(down)
        mlp.gate_up_proj, mlp.down_proj = gate_up, down
        mlp.gate_proj = mlp.up_proj = None

        attn = layer.self_attn
        projections = tuple(proj for proj in (attn.q_proj, attn.k_proj, attn.v_proj) if proj is not None)
        attn.qkv_widths = tuple(proj.weight.shape[0] for proj in projections)
        attn.qkv_proj = Int8Projection(projections)
        attn.o_proj = Int8Projection((attn.o_proj,))
        attn.q_proj = attn.k_proj = attn.v_proj = None

    def mlp_forward(module, value):
        if hasattr(module, "gate_up_proj"):
            gate, up = module.gate_up_proj(value).chunk(2, dim=-1)
            return module.down_proj(fused.gelu_mul(gate, up))
        return fused.mlp(module, value)

    def attention(attn, x, cos, sin, p):
        plan = p.plan
        query, key, value = project_attention(attn, x, cos, sin)
        if attn.is_sliding:
            whole = (key[0], value[0]) if plan.images is not None else None
            k, v = model._gather(key, value, plan.s_rows)
            out = sliding_attention.attend(query[0], k[0], v[0], plan.q_cu_t, plan.s_cu_t,
                                           plan.max_q, plan.max_s, attn.scaling, model.sliding[0])
            if whole is not None:
                out = vision.see_whole(out, query[0], whole[0], whole[1], plan.images, attn.scaling)
        elif isinstance(plan, PaddedPlan):
            out = global_attention.padded(query, key, value, plan.cu, plan.size, model.causal_global, attn.scaling)
        else:
            k, v = model._gather(key, value, plan.g_rows)
            out = model._global(query, k, v, plan.q_cu, plan.g_cu, plan.g_lens, model.causal_global, attn.scaling)
        return attn.o_proj(out.reshape(1, x.shape[1], -1))

    model.mlp = mlp_forward
    model.attention = attention
    torch.cuda.empty_cache()
