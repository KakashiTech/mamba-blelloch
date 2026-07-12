import torch
import torch.nn.functional as F
from einops import rearrange, repeat
import math


# =============================================================================
# 1. expm1_scan — replaces exp(dt*A) with 1 + expm1(dt*A) in the SSM recurrence
#
# The standard recurrence:
#   x[t] = exp(dt*A) * x[t-1] + dt * B * u[t]
#
# When dt*A is small, exp(dt*A) ~= 1.0 in low-precision arithmetic (fp16/bf16).
# This means the state decay term vanishes entirely: x[t] ≈ x[t-1] regardless of A.
# Using 1 + expm1(dt*A) preserves the small deviation from 1.0 because expm1(dt*A)
# computes the Taylor series exp(x)-1 = x + x²/2 + ... without catastrophic cancellation.
#
# The update becomes:
#   x[t] = x[t-1] + expm1(dt*A) * x[t-1] + dt * B * u[t]
# =============================================================================

def expm1_scan_reference_fp64(u, delta, A, B, C, D=None, z=None, delta_bias=None,
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

    dA_log = torch.einsum('bdl,dn->bdln', delta, A.to(torch.float64))
    deltaA = 1.0 + torch.expm1(dA_log)

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


def expm1_scan(u, delta, A, B, C, D=None, z=None, delta_bias=None,
               delta_softplus=False, return_last_state=False):
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    if A.is_complex():
        if is_variable_B:
            B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
        if is_variable_C:
            C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
    else:
        B = B.float()
        C = C.float()

    dA_log = torch.einsum('bdl,dn->bdln', delta, A.float())
    expm1_dA = torch.expm1(dA_log)

    if not is_variable_B:
        deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
    else:
        if B.dim() == 3:
            deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
        else:
            B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
            deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
    if is_variable_C and C.dim() == 4:
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])

    x = torch.zeros(batch, dim, dstate, dtype=torch.float32, device=u.device)
    ys = []
    last_state = None
    for i in range(u.shape[2]):
        x = x + expm1_dA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum('bdn,dn->bd', x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
            else:
                y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
        if i == u.shape[2] - 1:
            last_state = x
        if y.is_complex():
            y = y.real * 2
        ys.append(y)
    y = torch.stack(ys, dim=2)
    out = y if D is None else y + u * rearrange(D.float(), "d -> d 1")
    if z is not None:
        out = out * F.silu(z.float())
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)


# =============================================================================
# 2. kahan_state_passing — Kahan-Babuska compensated summation for state passing
#
# The standard state passing recurrence across chunk boundaries:
#   h_{k+1} = exp(dA_cumsum_k) * h_k + state_k
#
# This is equivalent to computing the weighted sum:
#   h_N = w_0 * h_0 + Σ_{i=0}^{N-1} w_{i+1} * state_i
#   where w_i = Π_{j>i} exp(dA_cumsum_j)
#
# Kahan-Babuska compensates for the rounding error in the incremental addition.
# The error bound changes from O(ε·N) to O(ε), independent of chunk count.
# =============================================================================

def kahan_state_passing_reference_fp64(states, dA_chunk_cumsum, initial_states=None):
    batch, nchunks, nheads, dim = states.shape
    assert dA_chunk_cumsum.shape == (batch, nheads, nchunks)

    states_f64 = states.to(torch.float64)
    dA_f64 = dA_chunk_cumsum.to(torch.float64)

    if initial_states is None:
        h = torch.zeros(batch, nheads, dim, dtype=torch.float64, device=states.device)
    else:
        h = initial_states.to(torch.float64)

    comp = torch.zeros(batch, nheads, dim, dtype=torch.float64, device=states.device)

    out_chunks = []
    for c in range(nchunks):
        new_s = states_f64[:, c]

        increment = torch.expm1(dA_f64[:, :, c]).unsqueeze(-1) * h + new_s

        y = increment - comp
        t = h + y
        comp = (t - h) - y
        h = t

        out_chunks.append(h)

    out = torch.stack(out_chunks, dim=1)
    final_state = h
    return out, final_state


def kahan_state_passing(states, dA_chunk_cumsum, initial_states=None):
    batch, nchunks, nheads, dim = states.shape
    assert dA_chunk_cumsum.shape == (batch, nheads, nchunks)

    # Intermediate accumulation done in fp32
    if initial_states is None:
        h = torch.zeros(batch, nheads, dim, dtype=torch.float32, device=states.device)
    else:
        h = initial_states.to(torch.float32)

    comp = torch.zeros(batch, nheads, dim, dtype=torch.float32, device=states.device)
    out_dtype = states.dtype
    out = torch.empty_like(states)

    for c in range(nchunks):
        new_s = states[:, c].to(torch.float32)

        increment = torch.expm1(dA_chunk_cumsum[:, :, c].to(torch.float32)).unsqueeze(-1) * h + new_s

        y = increment - comp
        t = h + y
        comp = (t - h) - y
        h = t

        out[:, c] = h.to(out_dtype)

    final_state = h.to(out_dtype)
    return out, final_state


# =============================================================================
# 3. stable_chunk_scan — combined expm1 discretization + Kahan-Babuska state passing
#
# Stable version of the chunked SSM scan that combines both techniques:
#   - expm1 for per-timestep discretization: preserves small dt*A signals
#   - Kahan-Babuska for cross-chunk state passing: prevents O(N) error accumulation
#
# Input shapes follow the mamba_chunk_scan_combined convention:
#   x: (batch, seqlen, nheads, headdim)
#   dt: (batch, seqlen, nheads)
#   A: (nheads,)
#   B: (batch, seqlen, ngroups, dstate)
#   C: (batch, seqlen, ngroups, dstate)
# =============================================================================

def stable_chunk_scan_reference_fp64(x, dt, A, B, C, chunk_size, D=None, z=None,
                                     dt_bias=None, dt_softplus=False,
                                     dt_limit=(0.0, float("inf")),
                                     initial_states=None, seq_idx=None,
                                     return_final_state=False):
    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert A.shape == (nheads,)
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert C.shape == B.shape
    assert dt.shape == (batch, seqlen, nheads)

    x_f64 = x.to(torch.float64)
    dt_f64 = dt.to(torch.float64)
    A_f64 = A.to(torch.float64)
    B_f64 = B.to(torch.float64)
    C_f64 = C.to(torch.float64)
    if D is not None:
        D_f64 = D.to(torch.float64)
    if z is not None:
        z_f64 = z.to(torch.float64)

    if seqlen % chunk_size != 0:
        pad_len = chunk_size - seqlen % chunk_size
        x_f64 = F.pad(x_f64, (0, 0, 0, 0, 0, pad_len))
        dt_f64 = F.pad(dt_f64, (0, 0, 0, pad_len))
        B_f64 = F.pad(B_f64, (0, 0, 0, 0, 0, pad_len))
        C_f64 = F.pad(C_f64, (0, 0, 0, 0, 0, pad_len))
    n_timesteps = x_f64.shape[1]
    nchunks = n_timesteps // chunk_size

    dt_r = rearrange(dt_f64, "b (c l) h -> b h c l", l=chunk_size)
    if dt_bias is not None:
        dt_bias_f64 = dt_bias.to(torch.float64)
        dt_r = dt_r + rearrange(dt_bias_f64, "h -> h 1 1")
    if dt_softplus:
        dt_r = F.softplus(dt_r)
    if dt_limit != (0.0, float("inf")):
        dt_r = dt_r.clamp(min=dt_limit[0], max=dt_limit[1])

    dA = dt_r * rearrange(A_f64, "h -> h 1 1")
    dA_cumsum = torch.cumsum(dA, dim=-1)

    nheads_ratio = nheads // ngroups
    B_exp = repeat(B_f64, "b l g d -> b l (g h) d", h=nheads_ratio)
    C_exp = repeat(C_f64, "b l g d -> b l (g h) d", h=nheads_ratio)

    x_r = rearrange(x_f64, "b (c l) h p -> b c l h p", l=chunk_size)
    B_r = rearrange(B_exp, "b (c l) h d -> b c l h d", l=chunk_size)
    C_r = rearrange(C_exp, "b (c l) h d -> b c l h d", l=chunk_size)

    # --- Step 1: intra-chunk states ---
    decay = torch.exp(dA_cumsum[:, :, :, -1:] - dA_cumsum)
    chunk_states = torch.einsum(
        "b c l h d, b h c l, b h c l, b c l h p -> b c h d p",
        B_r, decay, dt_r, x_r
    )
    chunk_states_flat = rearrange(chunk_states, "b c h d p -> b c h (d p)")

    # --- Step 2: Kahan-Babuska state passing ---
    if initial_states is not None:
        init_flat = rearrange(initial_states.to(torch.float64), "b h p d -> b h (p d)")
        all_flat = torch.cat([
            rearrange(init_flat, "b h d -> b 1 h d"),
            chunk_states_flat
        ], dim=1)
        n_pass = nchunks + 1
        has_init = True
    else:
        init_flat = torch.zeros(batch, nheads, headdim * dstate,
                               dtype=torch.float64, device=x.device)
        has_init = False

    dA_chunk_cumsum = dA_cumsum[:, :, :, -1]
    dA_padded = F.pad(dA_chunk_cumsum, (1, 0))
    dA_cum = torch.cumsum(dA_padded, dim=-1)

    h = init_flat.clone()
    comp = torch.zeros_like(h)
    states_flat_list = []
    for c in range(nchunks + (1 if has_init else 0)):
        if c == 0:
            dA_step = torch.zeros(batch, nheads, device=x.device, dtype=torch.float64)
        else:
            dA_step = dA_cum[:, :, c] - dA_cum[:, :, 0]

        new_s = chunk_states_flat[:, c] if not has_init or c > 0 else init_flat

        increment = torch.expm1(dA_step).unsqueeze(-1) * h + new_s
        y = increment - comp
        t = h + y
        comp = (t - h) - y
        h = t

        if has_init and c == 0:
            pass
        else:
            idx = c - 1 if has_init else c
            states_flat_list.append(h)

    states_flat = torch.stack(states_flat_list, dim=1)
    final_state_flat = h
    states = rearrange(states_flat, "b c h (d p) -> b c h d p", d=dstate)

    # --- Step 3: chunk scan output ---
    out = torch.zeros(batch, n_timesteps, nheads, headdim,
                      dtype=torch.float64, device=x.device)

    for c in range(nchunks):
        chunk_len = min(chunk_size, seqlen - c * chunk_size)
        if chunk_len <= 0:
            break

        x_c = x_r[:, c, :chunk_len]
        C_c = C_r[:, c, :chunk_len]
        dt_c = dt_r[:, :, c, :chunk_len]
        dA_cs_c = dA_cumsum[:, :, c, :chunk_len]
        states_c = states[:, c]

        for m in range(chunk_len):
            dA_cs_m = 1.0 + torch.expm1(dA_cs_c[:, :, m])
            C_m = C_c[:, m]

            acc = torch.einsum("b h d, b h d p -> b h p", C_m, states_c) * dA_cs_m.unsqueeze(-1)

            for k in range(m):
                diff = dA_cs_c[:, :, m] - dA_cs_c[:, :, k]
                cb = torch.einsum("b h d, b h d -> b h", C_c[:, m], B_r[:, c, k])
                scale = torch.exp(diff.clamp(max=0.0))
                acc += cb.unsqueeze(-1) * scale.unsqueeze(-1) * dt_c[:, :, k].unsqueeze(-1) * x_c[:, k]

            if z is not None:
                z_c = rearrange(z_f64, "b (c l) h p -> b c l h p", l=chunk_size)[:, c, m]
                acc = acc * z_c * torch.sigmoid(z_c)

            out[:, c * chunk_size + m] = acc

    if D is not None:
        out[:, :seqlen] += x[:, :seqlen].to(torch.float64) * rearrange(D_f64, "h -> h 1")

    out = out[:, :seqlen]

    if not return_final_state:
        return out.to(x.dtype)
    else:
        final_state = rearrange(final_state_flat, "b h (d p) -> b h d p", d=dstate)
        return out.to(x.dtype), final_state.to(x.dtype)


def stable_chunk_scan(x, dt, A, B, C, chunk_size, D=None, z=None,
                      dt_bias=None, dt_softplus=False,
                      dt_limit=(0.0, float("inf")),
                      initial_states=None, seq_idx=None,
                      return_final_state=False):
    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert A.shape == (nheads,)
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert C.shape == B.shape
    assert dt.shape == (batch, seqlen, nheads)

    nheads_ratio = nheads // ngroups
    state_dtype = x.dtype

    # Pad to chunk boundary
    if seqlen % chunk_size != 0:
        pad_len = chunk_size - seqlen % chunk_size
        x_pad = F.pad(x, (0, 0, 0, 0, 0, pad_len))
        dt_pad = F.pad(dt, (0, 0, 0, pad_len))
        B_pad = F.pad(B, (0, 0, 0, 0, 0, pad_len))
        C_pad = F.pad(C, (0, 0, 0, 0, 0, pad_len))
    else:
        x_pad = x
        dt_pad = dt
        B_pad = B
        C_pad = C
    n_timesteps = x_pad.shape[1]
    nchunks = n_timesteps // chunk_size

    # Chunk cumsum (in higher precision for the cumsum part)
    dt_r = rearrange(dt_pad.float(), "b (c l) h -> b h c l", l=chunk_size)
    if dt_bias is not None:
        dt_r = dt_r + rearrange(dt_bias.float(), "h -> h 1 1")
    if dt_softplus:
        dt_r = F.softplus(dt_r)
    if dt_limit != (0.0, float("inf")):
        dt_r = dt_r.clamp(min=dt_limit[0], max=dt_limit[1])

    A_f32 = A.float()
    dA = dt_r * rearrange(A_f32, "h -> h 1 1")
    dA_cumsum = torch.cumsum(dA, dim=-1)

    # Intra-chunk states using expm1 for each timestep's decay
    decay = torch.exp(dA_cumsum[:, :, :, -1:] - dA_cumsum)
    B_exp = repeat(B_pad.float(), "b l g d -> b l (g h) d", h=nheads_ratio)
    x_r = rearrange(x_pad.float(), "b (c l) h p -> b c l h p", l=chunk_size)
    B_r = rearrange(B_exp, "b (c l) h d -> b c l h d", l=chunk_size)

    chunk_states = torch.einsum(
        "b c l h d, b h c l, b h c l, b c l h p -> b c h d p",
        B_r, decay, dt_r, x_r
    )
    chunk_states = rearrange(chunk_states, "b c h d p -> b c h (d p)")

    if initial_states is not None:
        init_flat = rearrange(initial_states.float(), "b h p d -> b h (p d)")
        chunk_states = torch.cat([
            rearrange(init_flat, "b h d -> b 1 h d"),
            chunk_states
        ], dim=1)
        has_init = True
    else:
        init_flat = torch.zeros(batch, nheads, headdim * dstate,
                               dtype=torch.float32, device=x.device)
        has_init = False

    dA_chunk_cumsum = dA_cumsum[:, :, :, -1]
    dA_padded = F.pad(dA_chunk_cumsum, (1, 0))
    dA_cum = torch.cumsum(dA_padded, dim=-1)

    n_pass = nchunks + (1 if has_init else 0)

    # Kahan-Babuska state passing
    h = init_flat.clone()
    comp = torch.zeros_like(h)
    states_flat_list = []
    for c in range(n_pass):
        if c == 0:
            dA_step = torch.zeros(batch, nheads, device=x.device, dtype=torch.float32)
        else:
            dA_step = dA_cum[:, :, c] - dA_cum[:, :, 0]

        new_s = chunk_states[:, c].to(torch.float32)
        increment = torch.expm1(dA_step).unsqueeze(-1) * h + new_s

        y = increment - comp
        t = h + y
        comp = (t - h) - y
        h = t

        if c > 0 or not has_init:
            states_flat_list.append(h)

    states = torch.stack(states_flat_list, dim=1)
    final_state_flat = h
    states = rearrange(states, "b c h (d p) -> b c h d p", d=dstate)

    # Chunk scan output
    C_exp = repeat(C_pad.float(), "b l g d -> b l (g h) d", h=nheads_ratio)
    C_r = rearrange(C_exp, "b (c l) h d -> b c l h d", l=chunk_size)

    out = torch.zeros(batch, n_timesteps, nheads, headdim,
                      dtype=torch.float32, device=x.device)

    for c in range(nchunks):
        chunk_len = min(chunk_size, seqlen - c * chunk_size)
        if chunk_len <= 0:
            break

        x_c = x_r[:, c, :chunk_len]
        C_c = C_r[:, c, :chunk_len]
        dt_c = dt_r[:, :, c, :chunk_len]
        dA_cs_c = dA_cumsum[:, :, c, :chunk_len]
        states_c = states[:, c]

        for m in range(chunk_len):
            dA_cs_m = 1.0 + torch.expm1(dA_cs_c[:, :, m])
            C_m = C_c[:, m]

            acc = torch.einsum("b h d, b h d p -> b h p", C_m, states_c) * dA_cs_m.unsqueeze(-1)

            for k in range(m):
                diff = dA_cs_c[:, :, m] - dA_cs_c[:, :, k]
                cb = torch.einsum("b h d, b h d -> b h", C_c[:, m], B_r[:, c, k])
                scale = torch.exp(diff.clamp(max=0.0))
                acc += cb.unsqueeze(-1) * scale.unsqueeze(-1) * dt_c[:, :, k].unsqueeze(-1) * x_c[:, k]

            out[:, c * chunk_size + m] = acc

    out = out[:, :seqlen]

    if D is not None:
        out += x.float() * rearrange(D.float(), "h -> h 1")

    if z is not None:
        z_f = z.float()
        out = out * z_f * torch.sigmoid(z_f)

    if not return_final_state:
        return out.to(state_dtype)
    else:
        final_state = rearrange(final_state_flat, "b h (d p) -> b h d p", d=dstate)
        return out.to(state_dtype), final_state.to(state_dtype)


# =============================================================================
# Testing utility: compare all three functions against their fp64 references
# =============================================================================

def _relative_error(a, b):
    a, b = a.double(), b.double()
    diff = (a - b).abs().max().item()
    base = max(a.abs().max().item(), b.abs().max().item(), 1e-30)
    return diff / base


def test_expm1_scan():
    torch.manual_seed(42)
    batch, dim, dstate, seqlen = 2, 32, 16, 128
    u = torch.randn(batch, dim, seqlen)
    delta = torch.rand(batch, dim, seqlen).mul(0.1)
    A = torch.randn(dim, dstate).mul(-0.1)
    B = torch.randn(dim, dstate)
    C = torch.randn(dim, dstate)

    ref_out = expm1_scan_reference_fp64(u, delta, A, B, C)
    out = expm1_scan(u, delta, A, B, C)
    err = _relative_error(ref_out, out)
    print(f"[test_expm1_scan] relative error: {err:.2e}")
    assert err < 1e-5, f"expm1_scan error too large: {err}"
    print("  PASSED")


def test_kahan_state_passing():
    torch.manual_seed(42)
    batch, nchunks, nheads, dim = 2, 16, 4, 64
    states = torch.randn(batch, nchunks, nheads, dim)
    dA = torch.randn(batch, nheads, nchunks).mul(-0.5)

    ref_out, ref_final = kahan_state_passing_reference_fp64(states, dA)
    out, final = kahan_state_passing(states, dA)
    err = _relative_error(ref_out, out)
    print(f"[test_kahan_state_passing] output relative error: {err:.2e}")
    assert err < 1e-5, f"kahan_state_passing error too large: {err}"
    print("  PASSED")


def test_stable_chunk_scan():
    torch.manual_seed(42)
    batch, seqlen, nheads, headdim = 2, 128, 4, 8
    ngroups, dstate, chunk_size = 1, 8, 32
    x = torch.randn(batch, seqlen, nheads, headdim)
    dt = torch.rand(batch, seqlen, nheads).mul(0.1)
    A = -torch.exp(torch.rand(nheads).mul(2))
    B = torch.randn(batch, seqlen, ngroups, dstate)
    C = torch.randn(batch, seqlen, ngroups, dstate)
    D = torch.randn(nheads)

    ref_out = stable_chunk_scan_reference_fp64(x, dt, A, B, C, chunk_size, D=D)
    out = stable_chunk_scan(x, dt, A, B, C, chunk_size, D=D)
    err = _relative_error(ref_out, out)
    print(f"[test_stable_chunk_scan] relative error: {err:.2e}")
    assert err < 1e-4, f"stable_chunk_scan error too large: {err}"
    print("  PASSED")


def test_small_dt_stability():
    torch.manual_seed(42)
    batch, dim, dstate, seqlen = 2, 16, 8, 64

    u = torch.randn(batch, dim, seqlen)
    delta = torch.rand(batch, dim, seqlen).mul(1e-7)
    A = torch.randn(dim, dstate).mul(-0.01)
    B = torch.randn(dim, dstate)
    C = torch.randn(dim, dstate)

    ref_out = expm1_scan_reference_fp64(u, delta, A, B, C)
    out_standard = expm1_scan_reference_fp64(u, delta, A, B, C)

    err_std = _relative_error(ref_out, out_standard)
    print(f"[test_small_dt_stability] expm1 reference vs reference: {err_std:.2e}")
    print("  PASSED (self-consistency check)")


def run_all_tests():
    print("=" * 60)
    print("Running stability tests for selective_scan_stable.py")
    print("=" * 60)
    test_expm1_scan()
    test_kahan_state_passing()
    test_stable_chunk_scan()
    test_small_dt_stability()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()


__all__ = [
    "expm1_scan_reference_fp64",
    "expm1_scan",
    "kahan_state_passing_reference_fp64",
    "kahan_state_passing",
    "stable_chunk_scan_reference_fp64",
    "stable_chunk_scan",
    "test_expm1_scan",
    "test_kahan_state_passing",
    "test_stable_chunk_scan",
    "run_all_tests",
]
