"""Drop-in replacement for Mamba2's selective_scan_fn using the exact Blelloch prefix scan.

Usage:
    from mamba_ssm.ops.selective_scan_blelloch_interface import selective_scan_fn
    # Now use it exactly like the original Mamba2 selective_scan_fn
"""

import torch
from einops import rearrange, repeat

from selective_scan_blelloch import BlellochSSMFn


def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                      delta_softplus=False, return_last_state=False):
    """Exact Blelloch prefix scan replacement for Mamba2's selective_scan_fn.

    Args (Mamba2 format):
        u: (batch, dim, seqlen)
        delta: (batch, dim, seqlen)
        A: (dim, dstate) real, or (dim, dstate) complex
        B: (dim, dstate) or (batch, dstate, seqlen) or (batch, ngroups, dstate, seqlen)
        C: (dim, dstate) or (batch, dstate, seqlen) or (batch, ngroups, dstate, seqlen)
        D: (dim,)
        z: (batch, dim, seqlen)
        delta_bias: (dim,)

    Returns:
        out: (batch, dim, seqlen)
        last_state (optional): (batch, dim, dstate)
    """
    batch, dim, seqlen = u.shape
    orig_dtype = u.dtype

    u_f = u.float()
    delta_f = delta.float()

    is_complex = A.is_complex()
    dstate = A.shape[-1]
    if is_complex:
        from selective_scan_blelloch import _complex_to_real
        A_real = _complex_to_real(A).float()
        dstate = A_real.shape[-1]
    else:
        A_real = A.float()

    if A_real.dim() == 2:
        A_f = A_real
    elif A_real.dim() == 1:
        A_f = A_real.unsqueeze(-1).expand(-1, dstate)
    else:
        A_f = A_real

    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3

    if is_variable_B:
        if B.dim() == 4:
            ngroups = B.shape[1]
            B_t = rearrange(B.float(), "b g n l -> b l g n")
        else:
            ngroups = 1
            B_t = rearrange(B.float(), "b n l -> b l 1 n")
    else:
        ngroups = 1
        B_t = repeat(B.float(), "d n -> b l 1 n", b=batch, l=seqlen)

    if is_variable_C:
        if C.dim() == 4:
            C_t = rearrange(C.float(), "b g n l -> b l g n")
        else:
            C_t = rearrange(C.float(), "b n l -> b l 1 n")
    else:
        C_t = repeat(C.float(), "d n -> b l 1 n", b=batch, l=seqlen)

    if is_complex:
        nheads = dim
        headdim = 1
    else:
        nheads = dim
        headdim = 1

    x_t = rearrange(u_f, "b d l -> b l d").unsqueeze(-1)
    dt_t = rearrange(delta_f, "b d l -> b l d")

    out_t = BlellochSSMFn.apply(x_t, dt_t, A_f, B_t, C_t,
                                 D.view(-1, 1) if D is not None else None,
                                 rearrange(z.float(), "b d l -> b l d 1") if z is not None else None,
                                 delta_bias,
                                 delta_softplus,
                                 return_last_state=return_last_state,
                                 cu_seqlens=None,
                                 checkpoint_lvl=0)

    if return_last_state:
        out_t, last_state = out_t
        out = rearrange(out_t, "b l d 1 -> b d l").contiguous()
        last_state_out = last_state
        return out.to(orig_dtype), last_state_out

    out = rearrange(out_t, "b l d 1 -> b d l").contiguous()
    return out.to(orig_dtype)


def mamba_inner_fn(xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                   out_proj_weight, out_proj_bias,
                   A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
                   C_proj_bias=None, delta_softplus=True, checkpoint_lvl=0,
                   b_rms_weight=None, c_rms_weight=None, dt_rms_weight=None,
                   b_c_dt_rms_eps=1e-6):
    """Mamba2 inner function using Blelloch scan.

    This matches the signature of MambaInnerFn.forward from
    mamba_ssm.ops.selective_scan_interface, but uses the exact Blelloch scan.
    
    For a full integration, monkey-patch:
        mamba_ssm.ops.selective_scan_interface.selective_scan_fn = selective_scan_fn
    Or replace MambaInnerFn.forward's call to selective_scan_fn.
    """
    from mamba_ssm.ops.selective_scan_interface import _apply_dt_activation, MambaInnerFn

    batch, dim, seqlen = xz.shape
    dt_rank = delta_proj_weight.shape[1]

    dts = F.conv1d(xz[:, :dim], conv1d_weight, bias=conv1d_bias, padding=conv1d_weight.shape[-1] - 1, groups=dim)
    dts = dts[..., :seqlen]

    x_proj = F.linear(rearrange(dts, "b d l -> b l d"), x_proj_weight)
    x_proj = rearrange(x_proj, "b l d -> b d l")

    if x_proj_weight.shape[0] == dt_rank + 2 * dstate:
        delta = x_proj[:, :dt_rank]
        B_var = x_proj[:, dt_rank:dt_rank + dstate]
        C_var = x_proj[:, dt_rank + dstate:]
    else:
        delta = x_proj[:, :dt_rank]
        B_var = x_proj[:, dt_rank:dt_rank + dstate]
        C_var = x_proj[:, dt_rank + dstate:dt_rank + 2 * dstate]

    delta = F.linear(rearrange(delta, "b d l -> b l d"), delta_proj_weight)
    delta = rearrange(delta, "b l d -> b d l")

    if B is None:
        B = B_var
    if C is None:
        C = C_var

    z = xz[:, dim:]

    out = selective_scan_fn(
        u=xz[:, :dim],
        delta=delta,
        A=A,
        B=B,
        C=C,
        D=D,
        z=z,
        delta_bias=delta_bias,
        delta_softplus=delta_softplus,
        return_last_state=False,
    )

    out = F.linear(rearrange(out, "b d l -> b l d"), out_proj_weight, out_proj_bias)
    out = rearrange(out, "b l d -> b d l")
    return out
