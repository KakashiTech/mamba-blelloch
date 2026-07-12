import torch
import torch.nn.functional as F
import math
from einops import rearrange, repeat


def selective_scan_ref_fp64(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                            delta_softplus=False, return_last_state=False):
    u = u.to(torch.float64)
    delta = delta.to(torch.float64)
    if delta_bias is not None:
        delta = delta + delta_bias.to(torch.float64)[..., None]
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    B = B.to(torch.float64)
    C = C.to(torch.float64)
    A = A.to(torch.float64)

    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    if not is_variable_B:
        deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
    else:
        if B.dim() == 3:
            deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
        else:
            B = repeat(B, "b g n l -> b (g h) n l", h=dim // B.shape[1])
            deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
    if is_variable_C and C.dim() == 4:
        C = repeat(C, "b g n l -> b (g h) n l", h=dim // C.shape[1])

    x = torch.zeros(batch, dim, dstate, dtype=torch.float64, device=u.device)
    ys = []
    last_state = None
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum('bdn,dn->bd', x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
            else:
                y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
        if i == u.shape[2] - 1:
            last_state = x
        ys.append(y)
    y = torch.stack(ys, dim=2)
    out = y if D is None else y + u * rearrange(D.to(torch.float64), "d -> d 1")
    if z is not None:
        out = out * F.silu(z.to(torch.float64))
    if not return_last_state:
        return out
    return out, last_state


def chunk_cumsum_ref_fp64(dt, A, chunk_size, dt_bias=None, dt_softplus=False,
                          dt_limit=(0.0, float("inf"))):
    batch, seqlen, nheads = dt.shape
    assert A.shape == (nheads,)
    if dt_bias is not None:
        assert dt_bias.shape == (nheads,)

    dt = dt.to(torch.float64)
    A = A.to(torch.float64)
    if dt_bias is not None:
        dt_bias = dt_bias.to(torch.float64)

    if dt_bias is not None:
        dt = dt + dt_bias.unsqueeze(0).unsqueeze(1)
    if dt_softplus:
        dt = F.softplus(dt)
    dt = dt.clamp(min=dt_limit[0], max=dt_limit[1])

    nchunks = math.ceil(seqlen / chunk_size)
    if seqlen < nchunks * chunk_size:
        dt = F.pad(dt, (0, 0, 0, nchunks * chunk_size - seqlen))
    dt_out = rearrange(dt, "b (c l) h -> b h c l", l=chunk_size).contiguous()

    dA = dt_out * A.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    dA_cumsum = torch.cumsum(dA, dim=-1)

    return dA_cumsum, dt_out


def bmm_chunk_ref_fp64(a, b, chunk_size, seq_idx=None, causal=False):
    has_groups = a.dim() == 4
    if not has_groups:
        batch, seqlen, k = a.shape
        ngroups = 1
    else:
        batch, seqlen, ngroups, k = a.shape
    assert b.shape == a.shape

    a = a.to(torch.float64)
    b = b.to(torch.float64)

    nchunks = math.ceil(seqlen / chunk_size)
    if seqlen < nchunks * chunk_size:
        pad_len = nchunks * chunk_size - seqlen
        a = F.pad(a, (0, 0, 0, pad_len))
        b = F.pad(b, (0, 0, 0, pad_len))

    if not has_groups:
        a = rearrange(a, "b (c l) k -> b c l k", l=chunk_size)
        b = rearrange(b, "b (c l) k -> b c l k", l=chunk_size)
        out = torch.einsum("b c i k, b c j k -> b c i j", a, b)
    else:
        a = rearrange(a, "b (c l) g k -> b c l g k", l=chunk_size)
        b = rearrange(b, "b (c l) g k -> b c l g k", l=chunk_size)
        out = torch.einsum("b c i g k, b c j g k -> b c g i j", a, b)

    if causal:
        mask = torch.triu(torch.ones(chunk_size, chunk_size, device=out.device, dtype=torch.bool), diagonal=1)
        if not has_groups:
            out = out.masked_fill(mask.unsqueeze(0).unsqueeze(0), 0.0)
        else:
            out = out.masked_fill(mask.unsqueeze(0).unsqueeze(0).unsqueeze(0), 0.0)

    if seq_idx is not None:
        seq_idx_padded = seq_idx
        if seqlen < nchunks * chunk_size:
            seq_idx_padded = F.pad(seq_idx, (0, nchunks * chunk_size - seqlen), value=-1)
        seq_idx_chunked = rearrange(seq_idx_padded, "b (c l) -> b c l", l=chunk_size)
        if not has_groups:
            seq_i = seq_idx_chunked.unsqueeze(3)
            seq_j = seq_idx_chunked.unsqueeze(2)
            mask = seq_i == seq_j
            out = out * mask
        else:
            seq_i = seq_idx_chunked.unsqueeze(3).unsqueeze(4)
            seq_j = seq_idx_chunked.unsqueeze(2).unsqueeze(3)
            mask = seq_i == seq_j
            out = out * mask

    return out


def chunk_state_ref_fp64(B, x, dt, dA_cumsum, seq_idx=None):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0

    x = x.to(torch.float64)
    B = B.to(torch.float64)
    dt = dt.to(torch.float64)
    dA_cumsum = dA_cumsum.to(torch.float64)

    nheads_ratio = nheads // ngroups
    B_expanded = repeat(B, "b l g d -> b l (g h) d", h=nheads_ratio)

    if seqlen < nchunks * chunk_size:
        x = F.pad(x, (0, 0, 0, 0, 0, nchunks * chunk_size - seqlen))
        B_expanded = F.pad(B_expanded, (0, 0, 0, 0, 0, nchunks * chunk_size - seqlen))

    x = rearrange(x, "b (c l) h p -> b c l h p", l=chunk_size)
    B_expanded = rearrange(B_expanded, "b (c l) h d -> b c l h d", l=chunk_size)

    decay_states = torch.exp(dA_cumsum[:, :, :, -1:] - dA_cumsum)
    states = torch.einsum("b c l h d, b h c l, b h c l, b c l h p -> b c h p d",
                          B_expanded, decay_states, dt, x)

    return states


def state_passing_ref_fp64(states, dA_chunk_cumsum, initial_states=None):
    batch, nchunks, nheads, dim = states.shape
    assert dA_chunk_cumsum.shape == (batch, nheads, nchunks)

    states = states.to(torch.float64)
    dA_chunk_cumsum = dA_chunk_cumsum.to(torch.float64)

    if initial_states is None:
        initial_states = torch.zeros(batch, 1, nheads, dim, dtype=torch.float64, device=states.device)
    else:
        initial_states = initial_states.to(torch.float64).unsqueeze(1)
    states = torch.cat([initial_states, states], dim=1)

    dA_cumsum_padded = F.pad(dA_chunk_cumsum, (1, 0))
    dA_cumsum_cum = torch.cumsum(dA_cumsum_padded, dim=-1)

    n_total = dA_cumsum_cum.shape[-1]
    dt_chunk_segment_sum = dA_cumsum_cum[:, :, :, None] - dA_cumsum_cum[:, :, None, :]
    decay_chunk = torch.exp(dt_chunk_segment_sum)
    causal_mask = torch.tril(torch.ones(n_total, n_total, device=states.device, dtype=torch.bool), diagonal=0)
    decay_chunk = decay_chunk.masked_fill(~causal_mask, 0.0)

    out = torch.einsum("b h z c, b c h d -> b z h d", decay_chunk, states)
    final_states = out[:, -1]
    out = out[:, :-1]

    return out, final_states


def chunk_scan_ref_fp64(cb, x, dt, dA_cumsum, C, states, D=None, z=None, seq_idx=None):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = C.shape
    assert nheads % ngroups == 0
    assert cb.shape == (batch, nchunks, ngroups, chunk_size, chunk_size)

    x = x.to(torch.float64)
    C = C.to(torch.float64)
    dt = dt.to(torch.float64)
    dA_cumsum = dA_cumsum.to(torch.float64)
    cb = cb.to(torch.float64)
    states = states.to(torch.float64)
    if D is not None:
        D = D.to(torch.float64)
    if z is not None:
        z = z.to(torch.float64)

    nheads_ratio = nheads // ngroups
    C_expanded = repeat(C, "b l g d -> b l (g h) d", h=nheads_ratio)
    cb_expanded = repeat(cb, "b c g i j -> b c (g h) i j", h=nheads_ratio)

    if seqlen < nchunks * chunk_size:
        x = F.pad(x, (0, 0, 0, 0, 0, nchunks * chunk_size - seqlen))
        C_expanded = F.pad(C_expanded, (0, 0, 0, 0, 0, nchunks * chunk_size - seqlen))

    x = rearrange(x, "b (c l) h p -> b c l h p", l=chunk_size)
    C_expanded = rearrange(C_expanded, "b (c l) h d -> b c l h d", l=chunk_size)

    decay = torch.exp(dA_cumsum)

    out = torch.zeros(batch, nchunks * chunk_size, nheads, headdim, dtype=torch.float64, device=x.device)

    for c in range(nchunks):
        chunk_len = min(chunk_size, seqlen - c * chunk_size)
        if chunk_len <= 0:
            break
        x_c = x[:, c, :chunk_len]
        C_c = C_expanded[:, c, :chunk_len]
        dt_c = dt[:, :, c, :chunk_len]
        dA_cs_c = dA_cumsum[:, :, c, :chunk_len]
        cb_c = cb_expanded[:, c, :, :chunk_len, :chunk_len]
        states_c = states[:, c]

        for m in range(chunk_len):
            acc = torch.zeros(batch, nheads, headdim, dtype=torch.float64, device=x.device)
            dA_cs_m = dA_cs_c[:, :, m]

            C_m = C_c[:, m]
            acc += torch.einsum("b h d, b h p d -> b h p", C_m, states_c) * torch.exp(dA_cs_m).unsqueeze(-1)

            cb_m = cb_c[:, :, m, :chunk_len]
            dA_cs_k = dA_cs_c
            scale = torch.exp(torch.clamp(dA_cs_m.unsqueeze(-1) - dA_cs_k, max=0.0))
            cb_scaled = cb_m * scale * dt_c
            acc += torch.einsum("b h k, b k h p -> b h p", cb_scaled, x_c)

            if D is not None:
                x_res = x_c[:, m]
                if D.dim() == 2:
                    acc += x_res * D.unsqueeze(0)
                else:
                    acc += x_res * D.unsqueeze(0).unsqueeze(-1)

            if z is not None:
                z_c = rearrange(z, "b (c l) h p -> b c l h p", l=chunk_size)[:, c, m]
                acc = acc * F.silu(z_c)

            out[:, c * chunk_size + m] = acc

    if seqlen < nchunks * chunk_size:
        out = out[:, :seqlen]

    return out


__all__ = [
    "selective_scan_ref_fp64",
    "chunk_cumsum_ref_fp64",
    "bmm_chunk_ref_fp64",
    "chunk_state_ref_fp64",
    "state_passing_ref_fp64",
    "chunk_scan_ref_fp64",
]
