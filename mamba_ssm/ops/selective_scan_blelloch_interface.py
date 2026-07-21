"""Drop-in replacement for Mamba2's selective_scan_fn using the exact Blelloch prefix scan.

Usage:
    from mamba_ssm.ops.selective_scan_blelloch_interface import selective_scan_fn
    # Now use it exactly like the original Mamba2 selective_scan_fn
"""

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_blelloch import BlellochSSMFn, blelloch_ssm


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
        from mamba_ssm.ops.selective_scan_blelloch import _complex_to_real
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
                                 return_last_state,
                                 None,
                                 0,
                                 None)

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


def blelloch_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=None, z=None,
                                  dt_bias=None, initial_states=None, seq_idx=None,
                                  cu_seqlens=None, dt_softplus=False,
                                  dt_limit=(0.0, float("inf")),
                                  return_final_states=False,
                                  return_varlen_states=False, state_dtype=None):
    """Blelloch prefix scan drop-in replacement for Mamba2's chunk scan.

    Matches the signature of mamba_ssm.ops.triton.ssd_combined.mamba_chunk_scan_combined.
    The chunk_size parameter is accepted for API compatibility but ignored (Blelloch
    processes the full sequence in one pass rather than chunk-wise).

    Args:
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, seqlen, nheads)
        A: (nheads,) or (nheads, dstate)
        B: (batch, seqlen, ngroups, dstate)
        C: (batch, seqlen, ngroups, dstate)
        chunk_size: ignored (API compatibility)

    Returns:
        out: (batch, seqlen, nheads, headdim)
        (optional) final_state: (batch, nheads, headdim, dstate)
        (optional) varlen_states: (batch, nheads, headdim, dstate) if return_varlen_states
    """
    batch, seqlen, nheads, headdim = x.shape
    dstate = B.shape[-1]

    # 1. Apply dt_bias and dt_softplus
    dt_f = dt.float()
    if dt_bias is not None:
        dt_f = dt_f + dt_bias.float().unsqueeze(0).unsqueeze(1)
    if dt_softplus:
        dt_f = F.softplus(dt_f)

    # 2. Apply dt_limit
    if dt_limit is not None:
        dt_f = dt_f.clamp(min=dt_limit[0], max=dt_limit[1])

    # 3. Handle A: expand scalar-per-head to (nheads, dstate)
    A_f = A.float()
    if A_f.dim() == 1:
        A_f = A_f.unsqueeze(-1).expand(-1, dstate)

    # 4. Handle seq_idx masking
    if seq_idx is not None:
        seq_idx = seq_idx.int()

    # 5. Call blelloch_ssm with initial_states
    need_h = return_varlen_states and seq_idx is not None
    result = blelloch_ssm(
        x, dt_f.to(x.dtype), A_f, B, C,
        D=D, z=z, delta_bias=None, delta_softplus=False,
        return_last_state=return_final_states or return_varlen_states,
        cu_seqlens=cu_seqlens,
        checkpoint_lvl=0,
        initial_states=initial_states,
        seq_idx=seq_idx,
        state_dtype=state_dtype,
        return_h=need_h,
    )

    if return_final_states or return_varlen_states:
        if need_h:
            out, last_state, h_states = result
        else:
            out, last_state = result
            h_states = None
        final_state = last_state.permute(0, 3, 2, 1).contiguous()
        if return_final_states:
            return out, final_state
        if seq_idx is not None:
            seq_end = torch.cat([
                seq_idx[:, 1:] != seq_idx[:, :-1],
                torch.ones(batch, 1, dtype=torch.bool, device=seq_idx.device)
            ], dim=1)
            n_segments = seq_idx.max().item() + 1
            max_n = n_segments
            states = torch.zeros(max_n, batch, nheads, headdim, dstate,
                                 device=x.device, dtype=final_state.dtype)
            for i in range(max_n):
                mask = (seq_idx == i) & seq_end
                for b in range(batch):
                    pos = mask[b].nonzero()
                    if len(pos) > 0:
                        t = pos[-1].item()
                        if h_states is not None:
                            states[i, b] = h_states[b, t]
                        else:
                            states[i, b] = final_state[b]
            return out, states
        return out, final_state.unsqueeze(0)
    else:
        return (result,)


def blelloch_chunk_scan_combined_ref(x, dt, A, B, C, chunk_size, D=None, z=None,
                                      dt_bias=None, initial_states=None, seq_idx=None,
                                      cu_seqlens=None, dt_softplus=False,
                                      dt_limit=(0.0, float("inf")),
                                      return_final_states=False,
                                      return_varlen_states=False, state_dtype=None):
    """Pure-Python reference matching blelloch_chunk_scan_combined exactly.

    Used for numerical verification. Follows the same scan logic as
    ssd_minimal.ssd_minimal_discrete but with the exact Blelloch state update.
    """
    batch, seqlen, nheads, headdim = x.shape
    dstate = B.shape[-1]

    dt_f = dt.float()
    if dt_bias is not None:
        dt_f = dt_f + dt_bias.float().unsqueeze(0).unsqueeze(1)
    if dt_softplus:
        dt_f = F.softplus(dt_f)
    if dt_limit is not None:
        dt_f = dt_f.clamp(min=dt_limit[0], max=dt_limit[1])

    A_f = A.float()
    if A_f.dim() == 1:
        A_f = A_f.unsqueeze(-1).expand(-1, dstate)

    nheads_ratio = nheads // B.shape[2]
    B_exp = repeat(B.float(), "b l g d -> b l (g h) d", h=nheads_ratio)
    C_exp = repeat(C.float(), "b l g d -> b l (g h) d", h=nheads_ratio)

    if initial_states is not None:
        h = initial_states.float().permute(0, 1, 3, 2)  # (b, h, dstate, headdim)
    else:
        h = torch.zeros(batch, nheads, dstate, headdim, device=x.device, dtype=torch.float32)

    scale = torch.exp(dt_f.unsqueeze(-1) * A_f.unsqueeze(0).unsqueeze(1))

    ys = []
    for t in range(seqlen):
        if seq_idx is not None and t > 0 and seq_idx[0, t] != seq_idx[0, t - 1]:
            h = torch.zeros_like(h)
        drive = (dt_f[:, t, :].unsqueeze(-1).unsqueeze(-1) *
                 B_exp[:, t].unsqueeze(-1) *
                 x[:, t].float().unsqueeze(-2))
        h = scale[:, t].unsqueeze(-1) * h + drive
        y = torch.einsum("b h d p, b h d -> b h p", h, C_exp[:, t])
        ys.append(y)

    out = torch.stack(ys, dim=1)
    if D is not None:
        D_f = D.float()
        if D_f.dim() == 1:
            D_f = D_f.unsqueeze(-1)
        out = out + x.float() * D_f.unsqueeze(0).unsqueeze(1)
    if z is not None:
        z_f = z.float()
        if z_f.dim() == 3:
            z_f = z_f.unsqueeze(-1)
        out = out * F.silu(z_f)

    out = out.to(x.dtype)
    final_state = h.permute(0, 1, 3, 2).contiguous()  # (b, nheads, dstate, headdim) -> (b, nheads, headdim, dstate)

    if return_final_states:
        return out, final_state
    return out
