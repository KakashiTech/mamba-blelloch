import torch
import triton
import triton.language as tl

from einops import rearrange, repeat

from mamba_ssm.utils.determinism import autotune_configs


@triton.autotune(
    configs=autotune_configs([
        triton.Config({'BLOCK_SIZE': 64}),
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
    ]),
    key=['dim'],
)
@triton.jit
def _blelloch_scan_fwd_kernel(
    s_ptr, d_ptr, out_ptr,
    L, dim,
    stride_s_batch, stride_s_seqlen, stride_s_dim,
    stride_d_batch, stride_d_seqlen, stride_d_dim,
    stride_out_batch, stride_out_seqlen, stride_out_dim,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    pid_m = tl.program_id(axis=0)

    s_ptr += pid_b * stride_s_batch + pid_h * stride_s_seqlen * 0
    d_ptr += pid_b * stride_d_batch + pid_h * stride_d_seqlen * 0
    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_seqlen * 0

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    h = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for t in range(L):
        s = tl.load(s_ptr + t * stride_s_seqlen + offs_m * stride_s_dim,
                    mask=offs_m < dim, other=0.0).to(tl.float32)
        d = tl.load(d_ptr + t * stride_d_seqlen + offs_m * stride_d_dim,
                    mask=offs_m < dim, other=0.0).to(tl.float32)
        h = s * h + d
        out_ptrs = out_ptr + t * stride_out_seqlen + offs_m * stride_out_dim
        tl.store(out_ptrs, h, mask=offs_m < dim)


@triton.autotune(
    configs=autotune_configs([
        triton.Config({'BLOCK_SIZE': 64}),
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
    ]),
    key=['dim'],
)
@triton.jit
def _blelloch_scan_bwd_kernel(
    dout_ptr, s_ptr, out_ptr, ds_ptr, dd_ptr,
    L, dim,
    stride_dout_batch, stride_dout_seqlen, stride_dout_dim,
    stride_s_batch, stride_s_seqlen, stride_s_dim,
    stride_out_batch, stride_out_seqlen, stride_out_dim,
    stride_ds_batch, stride_ds_seqlen, stride_ds_dim,
    stride_dd_batch, stride_dd_seqlen, stride_dd_dim,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    pid_m = tl.program_id(axis=0)

    dout_ptr += pid_b * stride_dout_batch + pid_h * stride_dout_seqlen * 0
    s_ptr += pid_b * stride_s_batch + pid_h * stride_s_seqlen * 0
    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_seqlen * 0
    ds_ptr += pid_b * stride_ds_batch + pid_h * stride_ds_seqlen * 0
    dd_ptr += pid_b * stride_dd_batch + pid_h * stride_dd_seqlen * 0

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    g = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for t in range(L - 1, -1, -1):
        s = tl.load(s_ptr + t * stride_s_seqlen + offs_m * stride_s_dim,
                    mask=offs_m < dim, other=0.0).to(tl.float32)
        out = tl.load(out_ptr + t * stride_out_seqlen + offs_m * stride_out_dim,
                      mask=offs_m < dim, other=0.0).to(tl.float32)
        dout = tl.load(dout_ptr + t * stride_dout_seqlen + offs_m * stride_dout_dim,
                       mask=offs_m < dim, other=0.0).to(tl.float32)

        g = g + dout
        tl.store(ds_ptr + t * stride_ds_seqlen + offs_m * stride_ds_dim,
                 g * out, mask=offs_m < dim)
        tl.store(dd_ptr + t * stride_dd_seqlen + offs_m * stride_dd_dim,
                 g, mask=offs_m < dim)
        g = s * g


def blelloch_scan_fwd(s, d):
    batch, L, nheads, dstate, headdim = d.shape
    dim = dstate * headdim
    d_flat = rearrange(d, "b l h d p -> (b h) l (d p)")
    s_flat = rearrange(s, "b l h d -> (b h) l d")
    out = torch.empty_like(d_flat, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(dim, META['BLOCK_SIZE']), batch, nheads)
    with torch.cuda.device(d.device):
        _blelloch_scan_fwd_kernel[grid](
            s_flat, d_flat, out,
            L, dim,
            s_flat.stride(0), s_flat.stride(1), s_flat.stride(2),
            d_flat.stride(0), d_flat.stride(1), d_flat.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
        )
    out = rearrange(out, "(b h) l (d p) -> b l h d p", b=batch, h=nheads)
    return out


def blelloch_scan_bwd(dout, s, out):
    batch, L, nheads, dstate, headdim = out.shape
    dim = dstate * headdim
    dout_flat = rearrange(dout, "b l h d p -> (b h) l (d p)")
    s_flat = rearrange(s, "b l h d -> (b h) l d")
    out_flat = rearrange(out, "b l h d p -> (b h) l (d p)")
    ds = torch.empty_like(s_flat, dtype=torch.float32)
    dd = torch.empty_like(out_flat, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(dim, META['BLOCK_SIZE']), batch, nheads)
    with torch.cuda.device(dout.device):
        _blelloch_scan_bwd_kernel[grid](
            dout_flat, s_flat, out_flat, ds, dd,
            L, dim,
            dout_flat.stride(0), dout_flat.stride(1), dout_flat.stride(2),
            s_flat.stride(0), s_flat.stride(1), s_flat.stride(2),
            out_flat.stride(0), out_flat.stride(1), out_flat.stride(2),
            ds.stride(0), ds.stride(1), ds.stride(2),
            dd.stride(0), dd.stride(1), dd.stride(2),
        )
    ds = rearrange(ds, "(b h) l d -> b l h d", b=batch, h=nheads)
    dd = rearrange(dd, "(b h) l (d p) -> b l h d p", b=batch, h=nheads, d=dstate, p=headdim)
    return ds, dd


def blelloch_ssm_fwd(x, dt, A, B, C, D=None, z=None, delta_bias=None,
                     delta_softplus=False, return_last_state=False):
    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape
    nheads_ratio = nheads // ngroups

    from torch.nn import functional as F
    dt_f = dt.float()
    if delta_bias is not None:
        dt_f = dt_f + delta_bias.float().unsqueeze(0).unsqueeze(1)
    if delta_softplus:
        dt_f = F.softplus(dt_f)

    A_f = A.float()
    if A_f.dim() == 1:
        A_f = A_f.unsqueeze(-1).expand(-1, dstate)

    B_exp = repeat(B.float(), "b l g d -> b l (g h) d", h=nheads_ratio)
    C_exp = repeat(C.float(), "b l g d -> b l (g h) d", h=nheads_ratio)

    scale = torch.exp(dt_f.unsqueeze(-1) * A_f.unsqueeze(0).unsqueeze(1))
    drive = dt_f.unsqueeze(-1).unsqueeze(-1) * B_exp.unsqueeze(-1) * x.float().unsqueeze(-2)

    h = blelloch_scan_fwd(scale, drive)
    y = torch.einsum("b l h d p, b l h d -> b l h p", h, C_exp)

    if D is not None:
        y = y + x.float() * D.float().unsqueeze(0).unsqueeze(1)
    if z is not None:
        y = y * F.silu(z.float())

    out = y.to(x.dtype)
    if return_last_state:
        last_state = h[:, -1].permute(0, 2, 3, 1).to(x.dtype)
        return out, last_state
    return out
