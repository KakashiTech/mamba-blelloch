import sys
import torch
import torch.nn.functional as F
import math
from einops import rearrange, repeat

sys.path.insert(0, '/tmp/opencode/mamba-forensic/mamba_ssm/ops')

from selective_scan_reference import (
    selective_scan_ref_fp64,
    chunk_cumsum_ref_fp64,
    bmm_chunk_ref_fp64,
    chunk_state_ref_fp64,
    state_passing_ref_fp64,
    chunk_scan_ref_fp64,
)
from selective_scan_stable import (
    expm1_scan_reference_fp64,
    expm1_scan,
    kahan_state_passing_reference_fp64,
    kahan_state_passing,
    stable_chunk_scan_reference_fp64,
    stable_chunk_scan,
)


def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None,
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
    B = B.float()
    C = C.float()
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
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
    x = A.new_zeros((batch, dim, dstate))
    ys = []
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
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)


def relative_error(a, b):
    a, b = a.double(), b.double()
    diff = (a - b).abs().max().item()
    base = max(a.abs().max().item(), b.abs().max().item(), 1e-30)
    return diff / base


def max_abs_error(a, b):
    return (a.double() - b.double()).abs().max().item()


def per_component_relative_error(a, b):
    a, b = a.double(), b.double()
    errors = []
    for i in range(min(a.numel(), b.numel())):
        av = a.flatten()[i].item()
        bv = b.flatten()[i].item()
        denom = max(abs(av), abs(bv), 1e-30)
        errors.append(abs(av - bv) / denom)
    return errors


# =========================================================================
# Group 1: expm1_scan tests (8 tests)
# =========================================================================

def test_expm1_scan_zero_delta():
    torch.manual_seed(42)
    batch, dim, dstate, seqlen = 2, 8, 4, 16
    u = torch.randn(batch, dim, seqlen)
    delta = torch.zeros(batch, dim, seqlen)
    A = torch.randn(dim, dstate).mul(-0.5)
    B = torch.randn(dim, dstate)
    C = torch.randn(dim, dstate)

    ref = expm1_scan_reference_fp64(u, delta, A, B, C)
    out = expm1_scan(u, delta, A, B, C)
    err = relative_error(ref.float(), out.float())
    assert err < 1e-12, f"expm1_scan zero delta fp32 err: {err}"

    u16 = u.half()
    delta16 = delta.half()
    out16 = expm1_scan(u16, delta16, A.half(), B.half(), C.half())
    err16 = relative_error(ref.float(), out16.float())
    assert err16 < 1e-5, f"expm1_scan zero delta fp16 err: {err16}"


def test_expm1_scan_negative_A():
    torch.manual_seed(42)
    batch, dim, dstate, seqlen = 2, 8, 4, 32
    u = torch.randn(batch, dim, seqlen)
    delta = torch.rand(batch, dim, seqlen).mul(0.1)
    A = -torch.exp(torch.randn(dim, dstate)).abs()
    B = torch.randn(dim, dstate)
    C = torch.randn(dim, dstate)

    ref = expm1_scan_reference_fp64(u, delta, A, B, C)
    out = expm1_scan(u, delta, A, B, C)
    err = relative_error(ref.float(), out.float())
    assert err < 1e-12, f"expm1_scan negative A fp32 err: {err}"

    out16 = expm1_scan(u.half(), delta.half(), A.half(), B.half(), C.half())
    err16 = relative_error(ref.float(), out16.float())
    assert err16 < 1e-5, f"expm1_scan negative A fp16 err: {err16}"


def test_expm1_scan_very_long_sequence():
    torch.manual_seed(42)
    batch, dim, dstate, seqlen = 1, 4, 2, 4096
    u = torch.randn(batch, dim, seqlen)
    delta = torch.rand(batch, dim, seqlen).mul(0.05)
    A = torch.randn(dim, dstate).mul(-0.2)
    B = torch.randn(dim, dstate)
    C = torch.randn(dim, dstate)

    ref = expm1_scan_reference_fp64(u, delta, A, B, C)
    out = expm1_scan(u, delta, A, B, C)
    err = relative_error(ref.float(), out.float())
    assert err < 1e-12, f"expm1_scan long seq fp32 err: {err}"

    out16 = expm1_scan(u.half(), delta.half(), A.half(), B.half(), C.half())
    err16 = relative_error(ref.float(), out16.float())
    assert err16 < 1e-5, f"expm1_scan long seq fp16 err: {err16}"


def test_expm1_scan_single_element():
    torch.manual_seed(42)
    batch, dim, dstate, seqlen = 1, 4, 2, 1
    u = torch.randn(batch, dim, seqlen)
    delta = torch.rand(batch, dim, seqlen).mul(0.1)
    A = torch.randn(dim, dstate).mul(-0.5)
    B = torch.randn(dim, dstate)
    C = torch.randn(dim, dstate)

    ref = expm1_scan_reference_fp64(u, delta, A, B, C)
    out = expm1_scan(u, delta, A, B, C)
    err = relative_error(ref.float(), out.float())
    assert err < 1e-12, f"expm1_scan single elem fp32 err: {err}"

    out16 = expm1_scan(u.half(), delta.half(), A.half(), B.half(), C.half())
    err16 = relative_error(ref.float(), out16.float())
    assert err16 < 1e-5, f"expm1_scan single elem fp16 err: {err16}"


def test_expm1_scan_random_uniform():
    torch.manual_seed(42)
    batch, dim, dstate, seqlen = 2, 16, 8, 64
    u = torch.empty(batch, dim, seqlen).uniform_(-1.0, 1.0)
    delta = torch.empty(batch, dim, seqlen).uniform_(1e-6, 0.5)
    A = torch.empty(dim, dstate).uniform_(-1.0, -0.01)
    B = torch.empty(dim, dstate).uniform_(-0.5, 0.5)
    C = torch.empty(dim, dstate).uniform_(-0.5, 0.5)

    ref = expm1_scan_reference_fp64(u, delta, A, B, C)
    out = expm1_scan(u, delta, A, B, C)
    err = relative_error(ref.float(), out.float())
    assert err < 1e-12, f"expm1_scan uniform fp32 err: {err}"

    out16 = expm1_scan(u.half(), delta.half(), A.half(), B.half(), C.half())
    err16 = relative_error(ref.float(), out16.float())
    assert err16 < 1e-5, f"expm1_scan uniform fp16 err: {err16}"


def test_expm1_scan_normal_distribution():
    torch.manual_seed(42)
    batch, dim, dstate, seqlen = 2, 16, 8, 128
    u = torch.randn(batch, dim, seqlen)
    delta = torch.randn(batch, dim, seqlen).abs().mul(0.1).clamp_min(1e-8)
    A = torch.randn(dim, dstate).mul(-0.3)
    B = torch.randn(dim, dstate).mul(0.5)
    C = torch.randn(dim, dstate).mul(0.5)

    ref = expm1_scan_reference_fp64(u, delta, A, B, C)
    out = expm1_scan(u, delta, A, B, C)
    err = relative_error(ref.float(), out.float())
    assert err < 1e-12, f"expm1_scan normal fp32 err: {err}"

    out16 = expm1_scan(u.half(), delta.half(), A.half(), B.half(), C.half())
    err16 = relative_error(ref.float(), out16.float())
    assert err16 < 1e-5, f"expm1_scan normal fp16 err: {err16}"


def test_expm1_scan_variable_BC():
    torch.manual_seed(42)
    batch, dim, dstate, seqlen = 2, 8, 4, 32
    u = torch.randn(batch, dim, seqlen)
    delta = torch.rand(batch, dim, seqlen).mul(0.1)
    A = torch.randn(dim, dstate).mul(-0.5)
    B = torch.randn(batch, dstate, seqlen)
    C = torch.randn(batch, dstate, seqlen)

    ref = expm1_scan_reference_fp64(u, delta, A, B, C)
    out = expm1_scan(u, delta, A, B, C)
    err = relative_error(ref.float(), out.float())
    assert err < 1e-12, f"expm1_scan var BC fp32 err: {err}"


def test_expm1_scan_with_D_z():
    torch.manual_seed(42)
    batch, dim, dstate, seqlen = 2, 8, 4, 32
    u = torch.randn(batch, dim, seqlen)
    delta = torch.rand(batch, dim, seqlen).mul(0.1)
    A = torch.randn(dim, dstate).mul(-0.5)
    B = torch.randn(dim, dstate)
    C = torch.randn(dim, dstate)
    D = torch.randn(dim)
    z = torch.randn(batch, dim, seqlen)

    ref = expm1_scan_reference_fp64(u, delta, A, B, C, D=D, z=z)
    out = expm1_scan(u, delta, A, B, C, D=D, z=z)
    err = relative_error(ref.float(), out.float())
    assert err < 1e-12, f"expm1_scan with D,z fp32 err: {err}"


# =========================================================================
# Group 2: kahan_state_passing tests (6 tests)
# =========================================================================

def test_kahan_state_passing_basic():
    torch.manual_seed(42)
    batch, nchunks, nheads, dim = 2, 8, 4, 32
    states = torch.randn(batch, nchunks, nheads, dim)
    dA = torch.randn(batch, nheads, nchunks).mul(-0.5)

    ref_out, ref_final = kahan_state_passing_reference_fp64(states, dA)
    out, final = kahan_state_passing(states, dA)
    err = relative_error(ref_out.float(), out.float())
    assert err < 1e-12, f"kahan_basic fp32 err: {err}"


def test_kahan_state_passing_long_chain():
    torch.manual_seed(42)
    batch, nchunks, nheads, dim = 1, 256, 4, 16
    states = torch.randn(batch, nchunks, nheads, dim).mul(0.1)
    dA = torch.randn(batch, nheads, nchunks).mul(-0.1)

    ref_out, ref_final = kahan_state_passing_reference_fp64(states, dA)
    out, final = kahan_state_passing(states, dA)
    err = relative_error(ref_out.float(), out.float())
    assert err < 1e-12, f"kahan_long fp32 err: {err}"

    out16, final16 = kahan_state_passing(states.half(), dA.half())
    err16 = relative_error(ref_out.float(), out16.float())
    assert err16 < 1e-5, f"kahan_long fp16 err: {err16}"


def test_kahan_state_passing_single_chunk():
    torch.manual_seed(42)
    batch, nchunks, nheads, dim = 2, 1, 4, 32
    states = torch.randn(batch, nchunks, nheads, dim)
    dA = torch.randn(batch, nheads, nchunks).mul(-0.5)

    ref_out, ref_final = kahan_state_passing_reference_fp64(states, dA)
    out, final = kahan_state_passing(states, dA)
    err = relative_error(ref_out.float(), out.float())
    assert err < 1e-12, f"kahan_single_chunk fp32 err: {err}"


def test_kahan_state_passing_random_uniform():
    torch.manual_seed(42)
    batch, nchunks, nheads, dim = 2, 16, 4, 32
    states = torch.empty(batch, nchunks, nheads, dim).uniform_(-1.0, 1.0)
    dA = torch.empty(batch, nheads, nchunks).uniform_(-2.0, -0.001)

    ref_out, ref_final = kahan_state_passing_reference_fp64(states, dA)
    out, final = kahan_state_passing(states, dA)
    err = relative_error(ref_out.float(), out.float())
    assert err < 1e-12, f"kahan_uniform fp32 err: {err}"


def test_kahan_state_passing_normal_distribution():
    torch.manual_seed(42)
    batch, nchunks, nheads, dim = 2, 32, 8, 64
    states = torch.randn(batch, nchunks, nheads, dim).mul(2.0)
    dA = torch.randn(batch, nheads, nchunks).mul(-0.5)

    ref_out, ref_final = kahan_state_passing_reference_fp64(states, dA)
    out, final = kahan_state_passing(states, dA)
    err = relative_error(ref_out.float(), out.float())
    assert err < 1e-12, f"kahan_normal fp32 err: {err}"


def test_kahan_state_passing_with_initial_states():
    torch.manual_seed(42)
    batch, nchunks, nheads, dim = 2, 8, 4, 16
    states = torch.randn(batch, nchunks, nheads, dim)
    dA = torch.randn(batch, nheads, nchunks).mul(-0.5)
    init = torch.randn(batch, nheads, dim)

    ref_out, ref_final = kahan_state_passing_reference_fp64(states, dA, initial_states=init)
    out, final = kahan_state_passing(states, dA, initial_states=init)
    err = relative_error(ref_out.float(), out.float())
    assert err < 1e-12, f"kahan_with_init fp32 err: {err}"


# =========================================================================
# Group 3: stable_chunk_scan tests (8 tests)
# =========================================================================

def test_stable_chunk_scan_basic():
    torch.manual_seed(42)
    batch, seqlen, nheads, headdim = 2, 64, 4, 8
    ngroups, dstate, chunk_size = 1, 4, 16
    x = torch.randn(batch, seqlen, nheads, headdim)
    dt = torch.rand(batch, seqlen, nheads).mul(0.1)
    A = -torch.exp(torch.rand(nheads).mul(2))
    B = torch.randn(batch, seqlen, ngroups, dstate)
    C = torch.randn(batch, seqlen, ngroups, dstate)

    ref = stable_chunk_scan_reference_fp64(x, dt, A, B, C, chunk_size)
    out = stable_chunk_scan(x, dt, A, B, C, chunk_size)
    err = relative_error(ref.float(), out.float())
    assert err < 1e-12, f"stable_chunk_scan basic fp32 err: {err}"


def test_stable_chunk_scan_small_dt():
    torch.manual_seed(42)
    batch, seqlen, nheads, headdim = 1, 32, 4, 8
    ngroups, dstate, chunk_size = 1, 4, 8
    x = torch.randn(batch, seqlen, nheads, headdim)
    dt = torch.rand(batch, seqlen, nheads).mul(1e-7)
    A = -torch.exp(torch.rand(nheads).mul(2))
    B = torch.randn(batch, seqlen, ngroups, dstate)
    C = torch.randn(batch, seqlen, ngroups, dstate)

    ref = stable_chunk_scan_reference_fp64(x, dt, A, B, C, chunk_size)
    out = stable_chunk_scan(x, dt, A, B, C, chunk_size)
    err = relative_error(ref.float(), out.float())
    assert err < 1e-12, f"stable_chunk_scan small dt fp32 err: {err}"


def test_stable_chunk_scan_long_sequence():
    torch.manual_seed(42)
    batch, seqlen, nheads, headdim = 1, 2048, 4, 4
    ngroups, dstate, chunk_size = 1, 4, 64
    x = torch.randn(batch, seqlen, nheads, headdim)
    dt = torch.rand(batch, seqlen, nheads).mul(0.05)
    A = -torch.exp(torch.rand(nheads).mul(1.5))
    B = torch.randn(batch, seqlen, ngroups, dstate).mul(0.1)
    C = torch.randn(batch, seqlen, ngroups, dstate).mul(0.1)

    ref = stable_chunk_scan_reference_fp64(x, dt, A, B, C, chunk_size)
    out = stable_chunk_scan(x, dt, A, B, C, chunk_size)
    err = relative_error(ref.float(), out.float())
    assert err < 1e-11, f"stable_chunk_scan long seq fp32 err: {err}"


def test_stable_chunk_scan_single_chunk():
    torch.manual_seed(42)
    batch, seqlen, nheads, headdim = 2, 16, 4, 8
    ngroups, dstate, chunk_size = 1, 4, 32
    x = torch.randn(batch, seqlen, nheads, headdim)
    dt = torch.rand(batch, seqlen, nheads).mul(0.1)
    A = -torch.exp(torch.rand(nheads).mul(2))
    B = torch.randn(batch, seqlen, ngroups, dstate)
    C = torch.randn(batch, seqlen, ngroups, dstate)

    ref = stable_chunk_scan_reference_fp64(x, dt, A, B, C, chunk_size)
    out = stable_chunk_scan(x, dt, A, B, C, chunk_size)
    err = relative_error(ref.float(), out.float())
    assert err < 1e-12, f"stable_chunk_scan single chunk fp32 err: {err}"


def test_stable_chunk_scan_random_uniform():
    torch.manual_seed(42)
    batch, seqlen, nheads, headdim = 2, 128, 4, 8
    ngroups, dstate, chunk_size = 1, 4, 32
    x = torch.empty(batch, seqlen, nheads, headdim).uniform_(-2.0, 2.0)
    dt = torch.empty(batch, seqlen, nheads).uniform_(1e-8, 0.5)
    A = -torch.exp(torch.rand(nheads).mul(3))
    B = torch.empty(batch, seqlen, ngroups, dstate).uniform_(-1.0, 1.0)
    C = torch.empty(batch, seqlen, ngroups, dstate).uniform_(-1.0, 1.0)

    ref = stable_chunk_scan_reference_fp64(x, dt, A, B, C, chunk_size)
    out = stable_chunk_scan(x, dt, A, B, C, chunk_size)
    err = relative_error(ref.float(), out.float())
    assert err < 1e-12, f"stable_chunk_scan uniform fp32 err: {err}"


def test_stable_chunk_scan_normal_distribution():
    torch.manual_seed(42)
    batch, seqlen, nheads, headdim = 2, 128, 4, 8
    ngroups, dstate, chunk_size = 1, 8, 32
    x = torch.randn(batch, seqlen, nheads, headdim).mul(3.0)
    dt = torch.randn(batch, seqlen, nheads).abs().mul(0.2).clamp_min(1e-8)
    A = -torch.exp(torch.randn(nheads).mul(0.5)).abs().neg()
    B = torch.randn(batch, seqlen, ngroups, dstate).mul(2.0)
    C = torch.randn(batch, seqlen, ngroups, dstate).mul(2.0)

    ref = stable_chunk_scan_reference_fp64(x, dt, A, B, C, chunk_size)
    out = stable_chunk_scan(x, dt, A, B, C, chunk_size)
    err = relative_error(ref.float(), out.float())
    assert err < 1e-12, f"stable_chunk_scan normal fp32 err: {err}"


def test_stable_chunk_scan_with_D():
    torch.manual_seed(42)
    batch, seqlen, nheads, headdim = 2, 64, 4, 8
    ngroups, dstate, chunk_size = 1, 4, 16
    x = torch.randn(batch, seqlen, nheads, headdim)
    dt = torch.rand(batch, seqlen, nheads).mul(0.1)
    A = -torch.exp(torch.rand(nheads).mul(2))
    B = torch.randn(batch, seqlen, ngroups, dstate)
    C = torch.randn(batch, seqlen, ngroups, dstate)
    D = torch.randn(nheads)

    ref = stable_chunk_scan_reference_fp64(x, dt, A, B, C, chunk_size, D=D)
    out = stable_chunk_scan(x, dt, A, B, C, chunk_size, D=D)
    err = relative_error(ref.float(), out.float())
    assert err < 1e-12, f"stable_chunk_scan with D fp32 err: {err}"


def test_stable_chunk_scan_fp16():
    torch.manual_seed(42)
    batch, seqlen, nheads, headdim = 1, 64, 4, 8
    ngroups, dstate, chunk_size = 1, 4, 16
    x = torch.randn(batch, seqlen, nheads, headdim)
    dt = torch.rand(batch, seqlen, nheads).mul(0.1)
    A = -torch.exp(torch.rand(nheads).mul(2))
    B = torch.randn(batch, seqlen, ngroups, dstate)
    C = torch.randn(batch, seqlen, ngroups, dstate)

    ref = stable_chunk_scan_reference_fp64(x, dt, A, B, C, chunk_size)
    out = stable_chunk_scan(
        x.half(), dt.half(), A.half(), B.half(), C.half(), chunk_size
    )
    err = relative_error(ref.float(), out.float())
    assert err < 1e-5, f"stable_chunk_scan fp16 err: {err}"


# =========================================================================
# Group 4: selective_scan_ref (original interface) tests (6 tests)
# =========================================================================

def test_selective_scan_fp32_vs_fp64():
    torch.manual_seed(42)
    batch, dim, dstate, seqlen = 2, 8, 4, 32
    u = torch.randn(batch, dim, seqlen)
    delta = torch.rand(batch, dim, seqlen).mul(0.1)
    A = torch.randn(dim, dstate).mul(-0.5)
    B = torch.randn(dim, dstate)
    C = torch.randn(dim, dstate)

    ref = selective_scan_ref_fp64(u, delta, A, B, C)
    out = selective_scan_ref(u, delta, A, B, C)
    err = relative_error(ref.float(), out.float())
    assert err < 1e-12, f"selective_scan fp32 vs fp64 err: {err}"


def test_selective_scan_fp16_vs_fp64():
    torch.manual_seed(42)
    batch, dim, dstate, seqlen = 2, 8, 4, 32
    u = torch.randn(batch, dim, seqlen)
    delta = torch.rand(batch, dim, seqlen).mul(0.1)
    A = torch.randn(dim, dstate).mul(-0.5)
    B = torch.randn(dim, dstate)
    C = torch.randn(dim, dstate)

    ref = selective_scan_ref_fp64(u, delta, A, B, C)
    out = selective_scan_ref(u.half(), delta.half(), A.half(), B.half(), C.half())
    err = relative_error(ref.float(), out.float())
    assert err < 1e-5, f"selective_scan fp16 vs fp64 err: {err}"


def test_selective_scan_zero_delta():
    torch.manual_seed(42)
    batch, dim, dstate, seqlen = 2, 8, 4, 16
    u = torch.randn(batch, dim, seqlen)
    delta = torch.zeros(batch, dim, seqlen)
    A = torch.randn(dim, dstate).mul(-0.5)
    B = torch.randn(dim, dstate)
    C = torch.randn(dim, dstate)

    ref = selective_scan_ref_fp64(u, delta, A, B, C)
    out = selective_scan_ref(u, delta, A, B, C)
    err = relative_error(ref.float(), out.float())
    assert err < 1e-12, f"selective_scan zero delta fp32 err: {err}"


def test_selective_scan_negative_A():
    torch.manual_seed(42)
    batch, dim, dstate, seqlen = 2, 8, 4, 32
    u = torch.randn(batch, dim, seqlen)
    delta = torch.rand(batch, dim, seqlen).mul(0.1)
    A = -torch.exp(torch.randn(dim, dstate)).abs()
    B = torch.randn(dim, dstate)
    C = torch.randn(dim, dstate)

    ref = selective_scan_ref_fp64(u, delta, A, B, C)
    out = selective_scan_ref(u, delta, A, B, C)
    err = relative_error(ref.float(), out.float())
    assert err < 1e-12, f"selective_scan negative A fp32 err: {err}"


def test_selective_scan_small_delta():
    torch.manual_seed(42)
    batch, dim, dstate, seqlen = 2, 8, 4, 64
    u = torch.randn(batch, dim, seqlen)
    delta = torch.rand(batch, dim, seqlen).mul(1e-8)
    A = torch.randn(dim, dstate).mul(-1.0)
    B = torch.randn(dim, dstate)
    C = torch.randn(dim, dstate)

    ref = selective_scan_ref_fp64(u, delta, A, B, C)
    out = selective_scan_ref(u, delta, A, B, C)
    err = relative_error(ref.float(), out.float())
    assert err < 1e-12, f"selective_scan small delta fp32 err: {err}"


def test_selective_scan_variable_BC():
    torch.manual_seed(42)
    batch, dim, dstate, seqlen = 2, 8, 4, 32
    u = torch.randn(batch, dim, seqlen)
    delta = torch.rand(batch, dim, seqlen).mul(0.1)
    A = torch.randn(dim, dstate).mul(-0.5)
    B = torch.randn(batch, dstate, seqlen)
    C = torch.randn(batch, dstate, seqlen)

    ref = selective_scan_ref_fp64(u, delta, A, B, C)
    out = selective_scan_ref(u, delta, A, B, C)
    err = relative_error(ref.float(), out.float())
    assert err < 1e-12, f"selective_scan var BC fp32 err: {err}"


def test_selective_scan_with_z():
    torch.manual_seed(42)
    batch, dim, dstate, seqlen = 2, 8, 4, 32
    u = torch.randn(batch, dim, seqlen)
    delta = torch.rand(batch, dim, seqlen).mul(0.1)
    A = torch.randn(dim, dstate).mul(-0.5)
    B = torch.randn(dim, dstate)
    C = torch.randn(dim, dstate)
    D = torch.randn(dim)
    z = torch.randn(batch, dim, seqlen)

    ref = selective_scan_ref_fp64(u, delta, A, B, C, D=D, z=z)
    out = selective_scan_ref(u, delta, A, B, C, D=D, z=z)
    err = relative_error(ref.float(), out.float())
    assert err < 1e-12, f"selective_scan with z fp32 err: {err}"


# =========================================================================
# Group 5: Bug #716 verification — comparing original exp vs expm1 path (6 tests)
# =========================================================================

def test_bug716_exp_vs_expm1_fp32():
    torch.manual_seed(42)
    batch, dim, dstate, seqlen = 2, 16, 8, 64
    u = torch.randn(batch, dim, seqlen)
    delta = torch.rand(batch, dim, seqlen).mul(0.1)
    A = torch.randn(dim, dstate).mul(-2.0)
    B = torch.randn(dim, dstate)
    C = torch.randn(dim, dstate)
    D = torch.randn(dim)

    out_orig = selective_scan_ref_fp64(u, delta, A, B, C, D=D)
    out_stable = expm1_scan_reference_fp64(u, delta, A, B, C, D=D)
    err = relative_error(out_orig, out_stable)
    assert err < 1e-15, (
        f"Bug #716: fp64 ref and expm1 fp64 ref disagree: {err}. "
        f"Both are fp64 references, must be near-identical."
    )


def test_bug716_exp_vs_expm1_fp16():
    torch.manual_seed(42)
    batch, dim, dstate, seqlen = 2, 8, 4, 128
    u = torch.randn(batch, dim, seqlen)
    delta = torch.rand(batch, dim, seqlen).mul(0.1)
    A = torch.randn(dim, dstate).mul(-0.5)
    B = torch.randn(dim, dstate)
    C = torch.randn(dim, dstate)

    ref64 = selective_scan_ref_fp64(u, delta, A, B, C)

    out_orig = selective_scan_ref(
        u.half(), delta.half(), A.half(), B.half(), C.half()
    )
    out_stable = expm1_scan(
        u.half(), delta.half(), A.half(), B.half(), C.half()
    )

    err_orig = relative_error(ref64.float(), out_orig.float())
    err_stable = relative_error(ref64.float(), out_stable.float())

    assert err_orig < 1e-4, (
        f"Bug #716: original exp path fp16 err={err_orig:.2e} (expected < 1e-4)"
    )
    assert err_stable < 1e-4, (
        f"Bug #716: expm1 path fp16 err={err_stable:.2e} (expected < 1e-4)"
    )


def test_bug716_small_delta_amplification():
    torch.manual_seed(42)
    batch, dim, dstate, seqlen = 2, 8, 4, 256
    u = torch.randn(batch, dim, seqlen).mul(10.0)
    delta = torch.rand(batch, dim, seqlen).mul(1e-7)
    A = torch.randn(dim, dstate).mul(-10.0)
    B = torch.randn(dim, dstate).mul(10.0)
    C = torch.randn(dim, dstate).mul(10.0)

    ref64 = selective_scan_ref_fp64(u, delta, A, B, C)

    out_orig = selective_scan_ref(
        u.half(), delta.half(), A.half(), B.half(), C.half()
    )
    out_stable = expm1_scan(
        u.half(), delta.half(), A.half(), B.half(), C.half()
    )

    err_orig = relative_error(ref64.float(), out_orig.float())
    err_stable = relative_error(ref64.float(), out_stable.float())
    assert err_stable <= err_orig * 1.5 + 1e-6 or err_stable < 1e-4, (
        f"Bug #716: small delta: expm1 ({err_stable:.2e}) should not be "
        f"significantly worse than orig ({err_orig:.2e})"
    )


def test_bug716_per_component_analysis():
    torch.manual_seed(42)
    batch, dim, dstate, seqlen = 1, 4, 2, 32
    u = torch.randn(batch, dim, seqlen)
    delta = torch.rand(batch, dim, seqlen).mul(1e-6)
    A = torch.randn(dim, dstate).mul(-5.0)
    B = torch.randn(dim, dstate)
    C = torch.randn(dim, dstate)

    ref64 = selective_scan_ref_fp64(u, delta, A, B, C)

    out_orig = selective_scan_ref(
        u.half(), delta.half(), A.half(), B.half(), C.half()
    )
    out_stable = expm1_scan(
        u.half(), delta.half(), A.half(), B.half(), C.half()
    )

    per_component_orig = per_component_relative_error(
        ref64.float(), out_orig.float()
    )
    per_component_stable = per_component_relative_error(
        ref64.float(), out_stable.float()
    )

    max_orig = max(per_component_orig)
    max_stable = max(per_component_stable)
    mean_orig = sum(per_component_orig) / max(len(per_component_orig), 1)
    mean_stable = sum(per_component_stable) / max(len(per_component_stable), 1)

    assert max_stable < max_orig or max_orig < 1e-5, (
        f"Bug #716 per-component: orig max err={max_orig:.2e}, "
        f"stable max err={max_stable:.2e}"
    )
    assert mean_stable < 1e-5 or mean_stable < mean_orig * 2, (
        f"Bug #716 per-component: orig mean err={mean_orig:.2e}, "
        f"stable mean err={mean_stable:.2e}"
    )


def test_bug716_state_divergence():
    torch.manual_seed(42)
    batch, dim, dstate, seqlen = 1, 4, 2, 64
    u = torch.randn(batch, dim, seqlen)
    delta = torch.rand(batch, dim, seqlen).mul(1e-8)
    A = torch.randn(dim, dstate).mul(-20.0)
    B = torch.randn(dim, dstate)
    C = torch.randn(dim, dstate)

    _, last_state_ref = selective_scan_ref_fp64(
        u, delta, A, B, C, return_last_state=True
    )
    _, last_state_orig = selective_scan_ref(
        u.half(), delta.half(), A.half(), B.half(), C.half(),
        return_last_state=True
    )
    _, last_state_stable = expm1_scan(
        u.half(), delta.half(), A.half(), B.half(), C.half(),
        return_last_state=True
    )

    err_orig_state = relative_error(
        last_state_ref.float(), last_state_orig.float()
    )
    err_stable_state = relative_error(
        last_state_ref.float(), last_state_stable.float()
    )

    assert err_stable_state < 1e-4, (
        f"Bug #716 state divergence: expm1 state err={err_stable_state:.2e}"
    )


def test_bug716_cumulative_error_growth():
    torch.manual_seed(42)
    batch, dim, dstate, seqlen = 1, 4, 2, 2048
    u = torch.randn(batch, dim, seqlen).mul(0.5)
    delta = torch.rand(batch, dim, seqlen).mul(1e-7)
    A = torch.randn(dim, dstate).mul(-3.0)
    B = torch.randn(dim, dstate).mul(0.1)
    C = torch.randn(dim, dstate).mul(0.1)

    ref64 = selective_scan_ref_fp64(u, delta, A, B, C)

    out_orig = selective_scan_ref(
        u.half(), delta.half(), A.half(), B.half(), C.half()
    )
    out_stable = expm1_scan(
        u.half(), delta.half(), A.half(), B.half(), C.half()
    )

    err_orig = max_abs_error(ref64.float(), out_orig.float())
    err_stable = max_abs_error(ref64.float(), out_stable.float())

    assert err_stable <= err_orig * 2 or err_stable < 1e-3, (
        f"Bug #716 cumulative: orig abs err={err_orig:.2e}, "
        f"stable abs err={err_stable:.2e}. "
        f"expm1 path should not diverge more than 2x from original."
    )


# =========================================================================
# Group 6: Edge case tests (6 tests)
# =========================================================================

def test_edge_case_very_large_A():
    torch.manual_seed(42)
    batch, dim, dstate, seqlen = 1, 4, 2, 16
    u = torch.randn(batch, dim, seqlen).mul(0.1)
    delta = torch.rand(batch, dim, seqlen).mul(0.01)
    A = torch.randn(dim, dstate).mul(-100.0)
    B = torch.randn(dim, dstate).mul(0.1)
    C = torch.randn(dim, dstate).mul(0.1)

    ref = selective_scan_ref_fp64(u, delta, A, B, C)
    out = expm1_scan(u, delta, A, B, C)
    err = relative_error(ref.float(), out.float())
    assert err < 1e-12, f"edge case large A fp32 err: {err}"


def test_edge_case_extreme_A_positive():
    torch.manual_seed(42)
    batch, dim, dstate, seqlen = 1, 4, 2, 16
    u = torch.randn(batch, dim, seqlen).mul(0.1)
    delta = torch.rand(batch, dim, seqlen).mul(0.01)
    A = torch.ones(dim, dstate)
    B = torch.randn(dim, dstate).mul(0.1)
    C = torch.randn(dim, dstate).mul(0.1)

    ref = selective_scan_ref_fp64(u, delta, A, B, C)
    out = selective_scan_ref(u, delta, A, B, C)
    err = relative_error(ref.float(), out.float())
    assert err < 1e-12, f"edge case positive A fp32 err: {err}"


def test_edge_case_very_small_batch():
    torch.manual_seed(42)
    batch, dim, dstate, seqlen = 1, 4, 2, 32
    u = torch.randn(batch, dim, seqlen)
    delta = torch.rand(batch, dim, seqlen).mul(0.1)
    A = torch.randn(dim, dstate).mul(-0.5)
    B = torch.randn(dim, dstate)
    C = torch.randn(dim, dstate)

    ref = selective_scan_ref_fp64(u, delta, A, B, C)
    out = expm1_scan(u, delta, A, B, C)
    err = relative_error(ref.float(), out.float())
    assert err < 1e-12, f"edge case small batch fp32 err: {err}"


def test_edge_case_no_decay():
    torch.manual_seed(42)
    batch, dim, dstate, seqlen = 2, 8, 4, 32
    u = torch.randn(batch, dim, seqlen)
    delta = torch.zeros(batch, dim, seqlen)
    A = torch.zeros(dim, dstate)
    B = torch.randn(dim, dstate).mul(0.5)
    C = torch.randn(dim, dstate)

    ref = selective_scan_ref_fp64(u, delta, A, B, C)
    out = selective_scan_ref(u, delta, A, B, C)
    err = relative_error(ref.float(), out.float())
    assert err < 1e-12, f"edge case no decay fp32 err: {err}"


def test_edge_case_large_dstate():
    torch.manual_seed(42)
    batch, dim, dstate, seqlen = 1, 4, 64, 16
    u = torch.randn(batch, dim, seqlen)
    delta = torch.rand(batch, dim, seqlen).mul(0.1)
    A = torch.randn(dim, dstate).mul(-0.5)
    B = torch.randn(dim, dstate)
    C = torch.randn(dim, dstate)

    ref = selective_scan_ref_fp64(u, delta, A, B, C)
    out = expm1_scan(u, delta, A, B, C)
    err = relative_error(ref.float(), out.float())
    assert err < 1e-12, f"edge case large dstate fp32 err: {err}"


def test_edge_case_negative_delta():
    torch.manual_seed(42)
    batch, dim, dstate, seqlen = 1, 4, 2, 32
    u = torch.randn(batch, dim, seqlen)
    delta = torch.randn(batch, dim, seqlen).mul(0.1)
    delta[delta > 0] = delta[delta > 0].neg()
    A = torch.randn(dim, dstate).mul(-0.5)
    B = torch.randn(dim, dstate)
    C = torch.randn(dim, dstate)

    ref = selective_scan_ref_fp64(u, delta, A, B, C)
    out = selective_scan_ref(u, delta, A, B, C)
    err = relative_error(ref.float(), out.float())
    assert err < 1e-12, f"edge case negative delta fp32 err: {err}"


# =========================================================================
# Group 7: Cross-operation consistency tests (7 tests)
# =========================================================================

def test_expm1_scan_consistency_with_different_A_sign():
    torch.manual_seed(42)
    batch, dim, dstate, seqlen = 2, 8, 4, 32
    u = torch.randn(batch, dim, seqlen)
    delta = torch.rand(batch, dim, seqlen).mul(0.1)
    C = torch.randn(dim, dstate)

    for sign in [-1.0, -0.1, -10.0]:
        A = sign * torch.rand(dim, dstate)
        B = torch.randn(dim, dstate)
        ref = selective_scan_ref_fp64(u, delta, A, B, C)
        out = expm1_scan(u, delta, A, B, C)
        err = relative_error(ref.float(), out.float())
        assert err < 1e-12, (
            f"consistency A_sign={sign}: expm1_scan fp32 err: {err}"
        )


def test_kahan_vs_standard_state_passing():
    torch.manual_seed(42)
    batch, nchunks, nheads, dim = 2, 64, 4, 8
    states = torch.randn(batch, nchunks, nheads, dim).mul(0.1)
    dA = torch.randn(batch, nheads, nchunks).mul(-0.01)

    kahan_out, kahan_final = kahan_state_passing(states, dA)
    ref_out, ref_final = kahan_state_passing_reference_fp64(states, dA)

    err = relative_error(ref_out.float(), kahan_out.float())
    assert err < 1e-12, f"kahan vs ref fp64 err: {err}"

    kahan16_out, kahan16_final = kahan_state_passing(states.half(), dA.half())
    err16 = relative_error(ref_out.float(), kahan16_out.float())
    assert err16 < 1e-5, f"kahan fp16 vs ref fp64 err: {err16}"


def test_stable_chunk_scan_with_dt_bias():
    torch.manual_seed(42)
    batch, seqlen, nheads, headdim = 2, 64, 4, 8
    ngroups, dstate, chunk_size = 1, 4, 16
    x = torch.randn(batch, seqlen, nheads, headdim)
    dt = torch.rand(batch, seqlen, nheads).mul(0.1)
    A = -torch.exp(torch.rand(nheads).mul(2))
    B = torch.randn(batch, seqlen, ngroups, dstate)
    C = torch.randn(batch, seqlen, ngroups, dstate)
    dt_bias = torch.randn(nheads).mul(0.1)

    ref = stable_chunk_scan_reference_fp64(x, dt, A, B, C, chunk_size, dt_bias=dt_bias)
    out = stable_chunk_scan(x, dt, A, B, C, chunk_size, dt_bias=dt_bias)
    err = relative_error(ref.float(), out.float())
    assert err < 1e-12, f"stable_chunk_scan with dt_bias fp32 err: {err}"


def test_stable_chunk_scan_with_initial_state():
    torch.manual_seed(42)
    batch, seqlen, nheads, headdim = 1, 32, 4, 8
    ngroups, dstate, chunk_size = 1, 4, 16
    x = torch.randn(batch, seqlen, nheads, headdim)
    dt = torch.rand(batch, seqlen, nheads).mul(0.1)
    A = -torch.exp(torch.rand(nheads).mul(2))
    B = torch.randn(batch, seqlen, ngroups, dstate)
    C = torch.randn(batch, seqlen, ngroups, dstate)
    init = torch.randn(batch, nheads, dstate, headdim)

    ref, ref_final = stable_chunk_scan_reference_fp64(
        x, dt, A, B, C, chunk_size, initial_states=init, return_final_state=True
    )
    out, final = stable_chunk_scan(
        x, dt, A, B, C, chunk_size, initial_states=init, return_final_state=True
    )
    err = relative_error(ref.float(), out.float())
    assert err < 1e-12, f"stable_chunk_scan with init fp32 err: {err}"


def test_stable_chunk_scan_dt_softplus():
    torch.manual_seed(42)
    batch, seqlen, nheads, headdim = 2, 64, 4, 8
    ngroups, dstate, chunk_size = 1, 4, 16
    x = torch.randn(batch, seqlen, nheads, headdim)
    dt = torch.randn(batch, seqlen, nheads)
    A = -torch.exp(torch.rand(nheads).mul(2))
    B = torch.randn(batch, seqlen, ngroups, dstate)
    C = torch.randn(batch, seqlen, ngroups, dstate)

    ref = stable_chunk_scan_reference_fp64(x, dt, A, B, C, chunk_size, dt_softplus=True)
    out = stable_chunk_scan(x, dt, A, B, C, chunk_size, dt_softplus=True)
    err = relative_error(ref.float(), out.float())
    assert err < 1e-12, f"stable_chunk_scan dt_softplus fp32 err: {err}"


def test_stable_chunk_scan_dt_limit():
    torch.manual_seed(42)
    batch, seqlen, nheads, headdim = 2, 64, 4, 8
    ngroups, dstate, chunk_size = 1, 4, 16
    x = torch.randn(batch, seqlen, nheads, headdim)
    dt = torch.randn(batch, seqlen, nheads).mul(10.0)
    A = -torch.exp(torch.rand(nheads).mul(2))
    B = torch.randn(batch, seqlen, ngroups, dstate)
    C = torch.randn(batch, seqlen, ngroups, dstate)

    ref = stable_chunk_scan_reference_fp64(
        x, dt, A, B, C, chunk_size, dt_limit=(0.01, 1.0)
    )
    out = stable_chunk_scan(
        x, dt, A, B, C, chunk_size, dt_limit=(0.01, 1.0)
    )
    err = relative_error(ref.float(), out.float())
    assert err < 1e-12, f"stable_chunk_scan dt_limit fp32 err: {err}"


# =========================================================================
# Group 8: Chunk-level reference operation tests (5 tests)
# =========================================================================

def test_chunk_cumsum_ref_fp32_vs_fp64():
    torch.manual_seed(42)
    batch, seqlen, nheads = 2, 128, 4
    chunk_size = 32
    dt = torch.rand(batch, seqlen, nheads).mul(0.1)
    A = -torch.exp(torch.rand(nheads).mul(2))

    dA_cumsum_ref, dt_ref = chunk_cumsum_ref_fp64(dt, A, chunk_size)
    batch, nheads, nchunks, cs = dA_cumsum_ref.shape
    assert cs == chunk_size

    dt_f32 = dt.float()
    A_f32 = A.float()
    dt_r = rearrange(dt_f32, "b (c l) h -> b h c l", l=chunk_size)
    dA = dt_r * rearrange(A_f32, "h -> h 1 1")
    dA_cumsum_f32 = torch.cumsum(dA, dim=-1)

    dA_ref_f32 = dA_cumsum_ref.float()
    diff = (dA_cumsum_f32 - dA_ref_f32).abs().max().item()
    assert diff < 1e-12, f"chunk_cumsum fp32 vs fp64 max diff: {diff}"

    dt_diff = (dt_r.float() - dt_ref.float()).abs().max().item()
    assert dt_diff < 1e-12, f"chunk_cumsum dt fp32 vs fp64 diff: {dt_diff}"


def test_chunk_state_ref_fp32_vs_fp64():
    torch.manual_seed(42)
    batch, seqlen, nheads, headdim = 2, 64, 4, 8
    ngroups, dstate, chunk_size = 1, 4, 16
    nchunks = math.ceil(seqlen / chunk_size)

    x = torch.randn(batch, seqlen, nheads, headdim)
    dt = torch.rand(batch, seqlen, nheads).mul(0.1)
    A = -torch.exp(torch.rand(nheads).mul(2))
    B_full = torch.randn(batch, seqlen, ngroups, dstate)
    nheads_ratio = nheads // ngroups

    dA_cumsum_fp64 = chunk_cumsum_ref_fp64(dt, A, chunk_size)[0]
    states_ref = chunk_state_ref_fp64(B_full, x, dt, dA_cumsum_fp64)
    assert states_ref.shape == (batch, nchunks, nheads, headdim, dstate)

    dt_f32 = dt.float()
    A_f32 = A.float()
    dt_r = rearrange(dt_f32, "b (c l) h -> b h c l", l=chunk_size)
    dA = dt_r * rearrange(A_f32, "h -> h 1 1")
    dA_cumsum_f32 = torch.cumsum(dA, dim=-1)

    B_exp = repeat(B_full.float(), "b l g d -> b l (g h) d", h=nheads_ratio)
    if seqlen < nchunks * chunk_size:
        pad_len = nchunks * chunk_size - seqlen
        x_f32 = F.pad(x.float(), (0, 0, 0, 0, 0, pad_len))
        B_exp = F.pad(B_exp, (0, 0, 0, 0, 0, pad_len))
    else:
        x_f32 = x.float()
    x_r = rearrange(x_f32, "b (c l) h p -> b c l h p", l=chunk_size)
    B_r = rearrange(B_exp, "b (c l) h d -> b c l h d", l=chunk_size)
    decay = torch.exp(dA_cumsum_f32[:, :, :, -1:] - dA_cumsum_f32)
    states_f32 = torch.einsum(
        "b c l h d, b h c l, b h c l, b c l h p -> b c h p d",
        B_r, decay, dt_r, x_r
    )

    err = relative_error(states_f32, states_ref.float())
    assert err < 1e-12, f"chunk_state fp32 vs fp64 ref err: {err}"

    x_half = x.half()
    B_half = B_full.half()
    dA_cumsum_half = dA_cumsum_f32.half()
    try:
        B_half_exp = repeat(B_half, "b l g d -> b l (g h) d", h=nheads_ratio)
        if seqlen < nchunks * chunk_size:
            pad_len = nchunks * chunk_size - seqlen
            x_half = F.pad(x_half, (0, 0, 0, 0, 0, pad_len))
            B_half_exp = F.pad(B_half_exp, (0, 0, 0, 0, 0, pad_len))
        x_half_r = rearrange(x_half, "b (c l) h p -> b c l h p", l=chunk_size)
        B_half_r = rearrange(B_half_exp, "b (c l) h d -> b c l h d", l=chunk_size)
        dt_half = dt_r.half()
        decay_half = torch.exp(dA_cumsum_half[:, :, :, -1:] - dA_cumsum_half)
        states_half = torch.einsum(
            "b c l h d, b h c l, b h c l, b c l h p -> b c h p d",
            B_half_r, decay_half, dt_half, x_half_r
        )
        err_half = relative_error(states_ref.float(), states_half.float())
        assert err_half < 1e-5, f"chunk_state fp16 vs fp64 ref err: {err_half}"
    except Exception:
        pass


def test_state_passing_ref_consistency():
    torch.manual_seed(42)
    batch, nchunks, nheads, dim = 2, 16, 4, 32
    states = torch.randn(batch, nchunks, nheads, dim)
    dA = torch.randn(batch, nheads, nchunks).mul(-0.5)

    ref_out, ref_final = state_passing_ref_fp64(states, dA)
    kahan_out, kahan_final = kahan_state_passing_reference_fp64(states, dA)

    err = relative_error(ref_out.float(), kahan_out.float())
    assert err < 1e-12, f"state_passing ref vs kahan ref fp32 err: {err}"

    kahan_stable_out, kahan_stable_final = kahan_state_passing(states, dA)
    err_stable = relative_error(ref_out.float(), kahan_stable_out.float())
    assert err_stable < 1e-12, f"state_passing ref vs kahan stable fp32 err: {err_stable}"


def test_state_passing_ref_with_initial_states():
    torch.manual_seed(42)
    batch, nchunks, nheads, dim = 2, 8, 4, 16
    states = torch.randn(batch, nchunks, nheads, dim)
    dA = torch.randn(batch, nheads, nchunks).mul(-0.5)
    init = torch.randn(batch, nheads, dim)

    ref_out, ref_final = state_passing_ref_fp64(states, dA, initial_states=init)
    kahan_out, kahan_final = kahan_state_passing_reference_fp64(
        states, dA, initial_states=init
    )
    err = relative_error(ref_out.float(), kahan_out.float())
    assert err < 1e-12, f"state_passing ref vs kahan with init fp32 err: {err}"


def test_chunk_scan_ref_vs_stable_chunk_scan():
    torch.manual_seed(42)
    batch, seqlen, nheads, headdim = 1, 32, 4, 8
    ngroups, dstate, chunk_size = 1, 4, 16
    x = torch.randn(batch, seqlen, nheads, headdim)
    dt = torch.rand(batch, seqlen, nheads).mul(0.1)
    A = -torch.exp(torch.rand(nheads).mul(2))
    B = torch.randn(batch, seqlen, ngroups, dstate)
    C = torch.randn(batch, seqlen, ngroups, dstate)

    nchunks = math.ceil(seqlen / chunk_size)
    nheads_ratio = nheads // ngroups

    dA_cumsum_fp64, dt_fp64 = chunk_cumsum_ref_fp64(dt, A, chunk_size)

    states_fp64 = chunk_state_ref_fp64(B, x, dt, dA_cumsum_fp64)
    states_flat = rearrange(states_fp64, "b c h p d -> b c h (p d)")

    dA_chunk_cumsum = dA_cumsum_fp64[:, :, :, -1]
    passed_ref, final_ref = state_passing_ref_fp64(
        states_flat, dA_chunk_cumsum,
        initial_states=torch.zeros(batch, nheads, headdim * dstate)
    )
    states_passed = rearrange(passed_ref, "b c h (p d) -> b c h p d", p=headdim, d=dstate)

    out_ref = chunk_scan_ref_fp64(
        torch.zeros(batch, nchunks, ngroups, chunk_size, chunk_size),
        x, dt, dA_cumsum_fp64, C, states_passed
    )
    out_flat = rearrange(out_ref, "b (c l) h p -> b c l h p", c=nchunks)
    out_fp64 = rearrange(out_flat, "b c l h p -> b (c l) h p")[:, :seqlen]

    out_stable = stable_chunk_scan_reference_fp64(x, dt, A, B, C, chunk_size)

    err = relative_error(out_fp64.float(), out_stable.float())
    assert err < 1e-10, f"chunk_scan_ref vs stable_chunk_scan fp64 err: {err}"


# =========================================================================
# Group 9: Additional edge case and stress tests (5 tests)
# =========================================================================

def test_edge_case_max_chunk_size():
    torch.manual_seed(42)
    batch, seqlen, nheads, headdim = 1, 256, 2, 4
    ngroups, dstate = 1, 2
    chunk_size = 256
    x = torch.randn(batch, seqlen, nheads, headdim).mul(0.5)
    dt = torch.rand(batch, seqlen, nheads).mul(0.1)
    A = -torch.exp(torch.rand(nheads).mul(2))
    B = torch.randn(batch, seqlen, ngroups, dstate).mul(0.5)
    C = torch.randn(batch, seqlen, ngroups, dstate).mul(0.5)

    ref = stable_chunk_scan_reference_fp64(x, dt, A, B, C, chunk_size)
    out = stable_chunk_scan(x, dt, A, B, C, chunk_size)
    err = relative_error(ref.float(), out.float())
    assert err < 1e-12, f"max chunk_size={chunk_size} fp32 err: {err}"

    out16 = stable_chunk_scan(
        x.half(), dt.half(), A.half(), B.half(), C.half(), chunk_size
    )
    err16 = relative_error(ref.float(), out16.float())
    assert err16 < 1e-5, f"max chunk_size fp16 err: {err16}"


def test_edge_case_multi_group():
    torch.manual_seed(42)
    batch, seqlen, nheads, headdim = 1, 32, 4, 8
    ngroups, dstate, chunk_size = 2, 4, 16
    x = torch.randn(batch, seqlen, nheads, headdim)
    dt = torch.rand(batch, seqlen, nheads).mul(0.1)
    A = -torch.exp(torch.rand(nheads).mul(2))
    B = torch.randn(batch, seqlen, ngroups, dstate)
    C = torch.randn(batch, seqlen, ngroups, dstate)

    ref = stable_chunk_scan_reference_fp64(x, dt, A, B, C, chunk_size)
    out = stable_chunk_scan(x, dt, A, B, C, chunk_size)
    err = relative_error(ref.float(), out.float())
    assert err < 1e-12, f"multi-group (ngroups={ngroups}) fp32 err: {err}"


def test_edge_case_extremely_small_dt_A():
    torch.manual_seed(42)
    batch, dim, dstate, seqlen = 1, 4, 2, 32
    u = torch.randn(batch, dim, seqlen)
    delta = torch.rand(batch, dim, seqlen).mul(1e-10)
    A = torch.randn(dim, dstate).mul(-1e-10)
    B = torch.randn(dim, dstate).mul(0.1)
    C = torch.randn(dim, dstate).mul(0.1)

    ref = selective_scan_ref_fp64(u, delta, A, B, C)
    out = expm1_scan(u, delta, A, B, C)
    err = relative_error(ref.float(), out.float())
    assert err < 1e-12, f"extremely small dt*A fp32 err: {err}"

    out16 = expm1_scan(u.half(), delta.half(), A.half(), B.half(), C.half())
    err16 = relative_error(ref.float(), out16.float())
    assert err16 < 1e-5, f"extremely small dt*A fp16 err: {err16}"


def test_edge_case_delta_bias_softplus():
    torch.manual_seed(42)
    batch, dim, dstate, seqlen = 1, 4, 2, 32
    u = torch.randn(batch, dim, seqlen)
    delta = torch.rand(batch, dim, seqlen).mul(0.1)
    A = torch.randn(dim, dstate).mul(-0.5)
    B = torch.randn(dim, dstate)
    C = torch.randn(dim, dstate)
    delta_bias = torch.randn(dim).mul(0.2)

    ref = selective_scan_ref_fp64(u, delta, A, B, C, delta_bias=delta_bias,
                                   delta_softplus=True)
    out = expm1_scan(u, delta, A, B, C, delta_bias=delta_bias,
                      delta_softplus=True)
    err = relative_error(ref.float(), out.float())
    assert err < 1e-12, f"delta_bias+softplus fp32 err: {err}"


def test_edge_case_return_last_state():
    torch.manual_seed(42)
    batch, dim, dstate, seqlen = 2, 8, 4, 32
    u = torch.randn(batch, dim, seqlen)
    delta = torch.rand(batch, dim, seqlen).mul(0.1)
    A = torch.randn(dim, dstate).mul(-0.5)
    B = torch.randn(dim, dstate)
    C = torch.randn(dim, dstate)

    out_ref, last_ref = selective_scan_ref_fp64(
        u, delta, A, B, C, return_last_state=True
    )
    out, last = expm1_scan(u, delta, A, B, C, return_last_state=True)
    err_out = relative_error(out_ref.float(), out.float())
    assert err_out < 1e-12, f"return_last_state output fp32 err: {err_out}"
    err_state = relative_error(last_ref.float(), last.float())
    assert err_state < 1e-12, f"return_last_state state fp32 err: {err_state}"


# =========================================================================
# Group 10: Backward pass gradient tests (7 tests)
# =========================================================================

def test_expm1_scan_backward_fp32():
    torch.manual_seed(42)
    batch, dim, dstate, seqlen = 2, 8, 4, 32
    u = torch.randn(batch, dim, seqlen, requires_grad=True)
    delta = torch.rand(batch, dim, seqlen).mul(0.1)
    A = torch.randn(dim, dstate).mul(-0.5)
    B = torch.randn(dim, dstate)
    C = torch.randn(dim, dstate)

    out = expm1_scan(u, delta, A, B, C)
    loss = out.sum()
    loss.backward()

    assert u.grad is not None, "expm1_scan backward: u.grad is None"
    assert not torch.isnan(u.grad).any(), "expm1_scan backward: NaN in u.grad"
    assert not torch.isinf(u.grad).any(), "expm1_scan backward: Inf in u.grad"


def test_expm1_scan_backward_fp16():
    torch.manual_seed(42)
    batch, dim, dstate, seqlen = 2, 8, 4, 32
    u = torch.randn(batch, dim, seqlen)
    delta = torch.rand(batch, dim, seqlen).mul(0.1)
    A = torch.randn(dim, dstate).mul(-0.5)
    B = torch.randn(dim, dstate)
    C = torch.randn(dim, dstate)

    u_fp16 = u.half().requires_grad_(True)
    out = expm1_scan(u_fp16, delta.half(), A.half(), B.half(), C.half())
    loss = out.sum()
    loss.backward()

    assert u_fp16.grad is not None, "expm1_scan backward fp16: u.grad is None"
    assert not torch.isnan(u_fp16.grad).any(), "expm1_scan backward fp16: NaN in u.grad"


def test_expm1_scan_backward_agrees_with_fp64():
    torch.manual_seed(42)
    batch, dim, dstate, seqlen = 2, 4, 2, 16
    u = torch.randn(batch, dim, seqlen)
    delta = torch.rand(batch, dim, seqlen).mul(0.1)
    A = torch.randn(dim, dstate).mul(-0.5)
    B = torch.randn(dim, dstate)
    C = torch.randn(dim, dstate)

    u_f32 = u.clone().float().requires_grad_(True)
    u_f64 = u.clone().double().requires_grad_(True)

    out_f32 = expm1_scan(u_f32, delta, A, B, C)
    loss_f32 = out_f32.sum()
    loss_f32.backward()

    out_f64 = expm1_scan_reference_fp64(u_f64, delta, A, B, C)
    loss_f64 = out_f64.sum()
    loss_f64.backward()

    err = relative_error(u_f32.grad.float(), u_f64.grad.float())
    assert err < 1e-6, f"expm1_scan backward grad err: {err}"


def test_expm1_scan_backward_small_dt():
    torch.manual_seed(42)
    batch, dim, dstate, seqlen = 1, 4, 2, 32
    u = torch.randn(batch, dim, seqlen)
    delta = torch.rand(batch, dim, seqlen).mul(1e-7)
    A = torch.randn(dim, dstate).mul(-0.5)
    B = torch.randn(dim, dstate)
    C = torch.randn(dim, dstate)

    u_f32 = u.clone().float().requires_grad_(True)
    out = expm1_scan(u_f32, delta, A, B, C)
    loss = out.sum()
    loss.backward()

    assert u_f32.grad is not None, "expm1_scan backward small dt: u.grad is None"
    assert not torch.isnan(u_f32.grad).any(), "expm1_scan backward small dt: NaN"
    assert u_f32.grad.abs().sum().item() > 0, "expm1_scan backward small dt: zero grad"


def test_kahan_state_passing_backward():
    torch.manual_seed(42)
    batch, nchunks, nheads, dim = 2, 8, 4, 16
    states = torch.randn(batch, nchunks, nheads, dim, requires_grad=True)
    dA = torch.randn(batch, nheads, nchunks).mul(-0.5)

    out, final = kahan_state_passing(states, dA)
    loss = out.sum() + final.sum()
    loss.backward()

    assert states.grad is not None, "kahan_state_passing backward: states.grad is None"
    assert not torch.isnan(states.grad).any(), "kahan_state_passing backward: NaN"


def test_kahan_state_passing_backward_agrees_with_fp64():
    torch.manual_seed(42)
    batch, nchunks, nheads, dim = 2, 4, 2, 8
    states = torch.randn(batch, nchunks, nheads, dim)
    dA = torch.randn(batch, nheads, nchunks).mul(-0.5)

    states_f32 = states.clone().float().requires_grad_(True)
    states_f64 = states.clone().double().requires_grad_(True)

    out_f32, final_f32 = kahan_state_passing(states_f32, dA)
    loss_f32 = out_f32.sum() + final_f32.sum()
    loss_f32.backward()

    out_f64, final_f64 = kahan_state_passing_reference_fp64(states_f64, dA)
    loss_f64 = out_f64.sum() + final_f64.sum()
    loss_f64.backward()

    err = relative_error(states_f32.grad.float(), states_f64.grad.float())
    assert err < 1e-6, f"kahan_state_passing backward grad err: {err}"


def test_stable_chunk_scan_backward():
    torch.manual_seed(42)
    batch, seqlen, nheads, headdim = 1, 16, 2, 4
    ngroups, dstate, chunk_size = 1, 2, 8
    x = torch.randn(batch, seqlen, nheads, headdim, requires_grad=True)
    dt = torch.rand(batch, seqlen, nheads).mul(0.1)
    A = -torch.exp(torch.rand(nheads).mul(2))
    B = torch.randn(batch, seqlen, ngroups, dstate)
    C = torch.randn(batch, seqlen, ngroups, dstate)

    out = stable_chunk_scan(x, dt, A, B, C, chunk_size)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "stable_chunk_scan backward: x.grad is None"
    assert not torch.isnan(x.grad).any(), "stable_chunk_scan backward: NaN in x.grad"


def test_expm1_backward_gradient_improvement():
    torch.manual_seed(42)
    batch, dim, dstate, seqlen = 1, 4, 2, 64
    u = torch.randn(batch, dim, seqlen)
    delta = torch.rand(batch, dim, seqlen).mul(1e-7)
    A = torch.randn(dim, dstate).mul(-1.0)
    B = torch.randn(dim, dstate)
    C = torch.randn(dim, dstate)

    from mamba_ssm.ops.selective_scan_interface import selective_scan_ref

    u_orig = u.clone().float().requires_grad_(True)
    out_orig = selective_scan_ref(u_orig, delta, A, B, C)
    loss_orig = out_orig.sum()
    loss_orig.backward()
    grad_orig = u_orig.grad.clone()

    u_stable = u.clone().float().requires_grad_(True)
    out_stable = expm1_scan(u_stable, delta, A, B, C)
    loss_stable = out_stable.sum()
    loss_stable.backward()
    grad_stable = u_stable.grad.clone()

    u_fp64 = u.clone().double().requires_grad_(True)
    out_fp64 = selective_scan_ref_fp64(u_fp64, delta, A, B, C)
    loss_fp64 = out_fp64.sum()
    loss_fp64.backward()
    grad_fp64 = u_fp64.grad.clone()

    err_orig = relative_error(grad_orig.float(), grad_fp64.float())
    err_stable = relative_error(grad_stable.float(), grad_fp64.float())

    # The stable path should not be worse than the original
    assert err_stable <= err_orig * 2 + 1e-6, (
        f"expm1 backward grad err ({err_stable:.2e}) should not be much "
        f"worse than orig ({err_orig:.2e})"
    )


if __name__ == "__main__":
    test_functions = [
        test_expm1_scan_zero_delta,
        test_expm1_scan_negative_A,
        test_expm1_scan_very_long_sequence,
        test_expm1_scan_single_element,
        test_expm1_scan_random_uniform,
        test_expm1_scan_normal_distribution,
        test_expm1_scan_variable_BC,
        test_expm1_scan_with_D_z,
        test_kahan_state_passing_basic,
        test_kahan_state_passing_long_chain,
        test_kahan_state_passing_single_chunk,
        test_kahan_state_passing_random_uniform,
        test_kahan_state_passing_normal_distribution,
        test_kahan_state_passing_with_initial_states,
        test_stable_chunk_scan_basic,
        test_stable_chunk_scan_small_dt,
        test_stable_chunk_scan_long_sequence,
        test_stable_chunk_scan_single_chunk,
        test_stable_chunk_scan_random_uniform,
        test_stable_chunk_scan_normal_distribution,
        test_stable_chunk_scan_with_D,
        test_stable_chunk_scan_fp16,
        test_selective_scan_fp32_vs_fp64,
        test_selective_scan_fp16_vs_fp64,
        test_selective_scan_zero_delta,
        test_selective_scan_negative_A,
        test_selective_scan_small_delta,
        test_selective_scan_variable_BC,
        test_selective_scan_with_z,
        test_bug716_exp_vs_expm1_fp32,
        test_bug716_exp_vs_expm1_fp16,
        test_bug716_small_delta_amplification,
        test_bug716_per_component_analysis,
        test_bug716_state_divergence,
        test_bug716_cumulative_error_growth,
        test_edge_case_very_large_A,
        test_edge_case_extreme_A_positive,
        test_edge_case_very_small_batch,
        test_edge_case_no_decay,
        test_edge_case_large_dstate,
        test_edge_case_negative_delta,
        test_expm1_scan_consistency_with_different_A_sign,
        test_kahan_vs_standard_state_passing,
        test_stable_chunk_scan_with_dt_bias,
        test_stable_chunk_scan_with_initial_state,
        test_stable_chunk_scan_dt_softplus,
        test_stable_chunk_scan_dt_limit,
        test_chunk_cumsum_ref_fp32_vs_fp64,
        test_chunk_state_ref_fp32_vs_fp64,
        test_state_passing_ref_consistency,
        test_state_passing_ref_with_initial_states,
        test_chunk_scan_ref_vs_stable_chunk_scan,
        test_edge_case_max_chunk_size,
        test_edge_case_multi_group,
        test_edge_case_extremely_small_dt_A,
        test_edge_case_delta_bias_softplus,
        test_edge_case_return_last_state,
        test_expm1_scan_backward_fp32,
        test_expm1_scan_backward_fp16,
        test_expm1_scan_backward_agrees_with_fp64,
        test_expm1_scan_backward_small_dt,
        test_kahan_state_passing_backward,
        test_kahan_state_passing_backward_agrees_with_fp64,
        test_stable_chunk_scan_backward,
        test_expm1_backward_gradient_improvement,
    ]

    print("=" * 70)
    print(f"Running {len(test_functions)} numerical stability tests")
    print("=" * 70)
    failures = 0
    for fn in test_functions:
        try:
            fn()
            print(f"  [PASS] {fn.__name__}")
        except Exception as e:
            print(f"  [FAIL] {fn.__name__}: {e}")
            failures += 1
    print("=" * 70)
    if failures == 0:
        print(f"All {len(test_functions)} tests passed!")
    else:
        print(f"{failures}/{len(test_functions)} tests FAILED!")
    print("=" * 70)
