"""
Blelloch parallel prefix scan for SSM recurrence h[t] = s[t]*h[t-1] + d[t].

combine((s1,d1),(s2,d2)) = (s1*s2, s2*d1 + d2)

Up-sweep (reduce) and down-sweep (exclusive) as separate kernel layers.
Uses tl.arange() for vectorized HEADDIM operations.
"""
import math
import torch
import triton
import triton.language as tl
from einops import rearrange, repeat


@triton.jit
def _up_sweep_kernel(
    s_ptr, d_ptr,
    step,
    stride_s_batch, stride_s_seq, stride_s_head, stride_s_dstate,
    stride_d_batch, stride_d_seq, stride_d_head, stride_d_dstate, stride_d_headdim,
    HEADDIM: tl.constexpr,
):
    pid_pair = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_n = tl.program_id(3)

    right_idx = (pid_pair + 1) * step - 1
    left_idx = right_idx - step // 2

    base_s = pid_b * stride_s_batch + pid_h * stride_s_head + pid_n * stride_s_dstate
    base_d = pid_b * stride_d_batch + pid_h * stride_d_head + pid_n * stride_d_dstate

    s_left = tl.load(s_ptr + base_s + left_idx * stride_s_seq)
    s_r_old = tl.load(s_ptr + base_s + right_idx * stride_s_seq)
    s_new = s_left * s_r_old
    tl.store(s_ptr + base_s + right_idx * stride_s_seq, s_new)

    off_l = base_d + left_idx * stride_d_seq
    off_r = base_d + right_idx * stride_d_seq
    h = tl.arange(0, HEADDIM)
    d_left = tl.load(d_ptr + off_l + h * stride_d_headdim)
    d_r_old = tl.load(d_ptr + off_r + h * stride_d_headdim)
    d_new = s_r_old * d_left + d_r_old
    tl.store(d_ptr + off_r + h * stride_d_headdim, d_new)


@triton.jit
def _down_sweep_kernel(
    s_ptr, d_ptr,
    step,
    stride_s_batch, stride_s_seq, stride_s_head, stride_s_dstate,
    stride_d_batch, stride_d_seq, stride_d_head, stride_d_dstate, stride_d_headdim,
    HEADDIM: tl.constexpr,
):
    pid_pair = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_n = tl.program_id(3)

    right_idx = (pid_pair + 1) * step - 1
    left_idx = right_idx - step // 2

    base_s = pid_b * stride_s_batch + pid_h * stride_s_head + pid_n * stride_s_dstate
    base_d = pid_b * stride_d_batch + pid_h * stride_d_head + pid_n * stride_d_dstate

    sl_old = tl.load(s_ptr + base_s + left_idx * stride_s_seq)
    pr_s = tl.load(s_ptr + base_s + right_idx * stride_s_seq)

    h = tl.arange(0, HEADDIM)
    off_l = base_d + left_idx * stride_d_seq
    off_r = base_d + right_idx * stride_d_seq
    dl_old = tl.load(d_ptr + off_l + h * stride_d_headdim)
    pr_d = tl.load(d_ptr + off_r + h * stride_d_headdim)

    tl.store(d_ptr + off_l + h * stride_d_headdim, pr_d)
    tl.store(d_ptr + off_r + h * stride_d_headdim, pr_s * dl_old + pr_d)
    tl.store(s_ptr + base_s + left_idx * stride_s_seq, pr_s)
    tl.store(s_ptr + base_s + right_idx * stride_s_seq, sl_old * pr_s)


def blelloch_scan_fwd(s, d):
    """
    Blelloch exclusive prefix scan.
    s: (batch, seqlen, nheads, dstate)
    d: (batch, seqlen, nheads, dstate, headdim)
    Returns d_prefix: (batch, seqlen, nheads, dstate, headdim)
    """
    batch, seqlen, nheads, dstate = s.shape
    headdim = d.shape[-1]
    N = 1
    while N < seqlen: N <<= 1

    # Working buffers (padded to power of 2)
    s_buf = torch.empty(batch, N, nheads, dstate, device=s.device, dtype=torch.float32)
    d_buf = torch.zeros(batch, N, nheads, dstate, headdim, device=s.device, dtype=torch.float32)
    s_buf[:, :seqlen] = s.float()
    s_buf[:, seqlen:] = 1.0
    d_buf[:, :seqlen] = d.float()

    cfg = {'HEADDIM': headdim}

    # Up-sweep
    step = 2
    while step <= N:
        n_pairs = N // step
        if n_pairs > 0:
            _up_sweep_kernel[(n_pairs, batch, nheads, dstate), cfg](
                s_buf, d_buf, step,
                s_buf.stride(0), s_buf.stride(1), s_buf.stride(2), s_buf.stride(3),
                d_buf.stride(0), d_buf.stride(1), d_buf.stride(2), d_buf.stride(3), d_buf.stride(4),
            )
        step <<= 1

    # Set root to identity (last position)
    d_buf[:, -1] = 0.0

    # Down-sweep
    step_val = N.bit_length() - 2
    while step_val >= 0:
        cs = 1 << (step_val + 1)
        n_pairs = N // cs
        if n_pairs > 0:
            _down_sweep_kernel[(n_pairs, batch, nheads, dstate), cfg](
                s_buf, d_buf, cs,
                s_buf.stride(0), s_buf.stride(1), s_buf.stride(2), s_buf.stride(3),
                d_buf.stride(0), d_buf.stride(1), d_buf.stride(2), d_buf.stride(3), d_buf.stride(4),
            )
        step_val -= 1

    return d_buf[:, :seqlen]


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

    d_prefix = blelloch_scan_fwd(scale, drive)
    h_incl = scale.unsqueeze(-1) * d_prefix + drive
    y = torch.einsum("b l h n d, b l h n -> b l h d", h_incl, C_exp)

    if D is not None:
        y = y + x.float() * D.float().unsqueeze(0).unsqueeze(1)
    if z is not None:
        y = y * F.silu(z.float())

    out = y.to(x.dtype)
    if return_last_state:
        last_state = h_incl[:, -1].permute(0, 2, 3, 1).to(x.dtype)
        return out, last_state
    return out
