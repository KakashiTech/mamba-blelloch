"""Tests for Blelloch prefix scan integration with Mamba2 SSD API.

Compares blelloch_chunk_scan_combined against:
  1. blelloch_chunk_scan_combined_ref (pure-Python reference)
  2. mamba_chunk_scan_combined (Triton, if CUDA available)
"""

import torch
import torch.nn.functional as F
import pytest

from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_blelloch_interface import (
    blelloch_chunk_scan_combined,
    blelloch_chunk_scan_combined_ref,
)

try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    HAS_TRITON = torch.cuda.is_available()
except ImportError:
    mamba_chunk_scan_combined = None
    HAS_TRITON = False


DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("seqlen", [16, 32, 64])
@pytest.mark.parametrize("nheads", [2, 4])
@pytest.mark.parametrize("headdim", [8, 16])
@pytest.mark.parametrize("dstate", [4, 8])
def test_forward_ref(device, batch, seqlen, nheads, headdim, dstate):
    """Blelloch forward matches pure-Python reference."""
    torch.manual_seed(42)
    ngroups = 1

    x = torch.randn(batch, seqlen, nheads, headdim, device=device)
    dt = torch.rand(batch, seqlen, nheads, device=device).mul(0.1)
    A = -torch.exp(torch.rand(nheads, dstate, device=device).mul(2))
    B = torch.randn(batch, seqlen, ngroups, dstate, device=device)
    C = torch.randn(batch, seqlen, ngroups, dstate, device=device)
    D = torch.randn(nheads, headdim, device=device)
    z = torch.randn(batch, seqlen, nheads, headdim, device=device)
    dt_bias = torch.randn(nheads, device=device)

    out = blelloch_chunk_scan_combined(
        x, dt, A, B, C, chunk_size=64, D=D, z=z,
        dt_bias=dt_bias, dt_softplus=True,
    )[0]

    ref = blelloch_chunk_scan_combined_ref(
        x, dt, A, B, C, chunk_size=64, D=D, z=z,
        dt_bias=dt_bias, dt_softplus=True,
    )

    err = (out.float() - ref.float()).abs().max().item()
    assert err < 1e-4, f"Forward mismatch: max_err={err:.2e}"


@pytest.mark.parametrize("device", DEVICES)
def test_backward(device):
    """Blelloch backward matches reference backward."""
    torch.manual_seed(42)
    batch, seqlen, nheads, headdim = 2, 16, 2, 8
    dstate, ngroups = 4, 1

    x = torch.randn(batch, seqlen, nheads, headdim, device=device, requires_grad=True)
    dt = torch.rand(batch, seqlen, nheads, device=device)
    dt.mul_(0.1)
    dt.requires_grad_(True)
    A = -torch.exp(torch.rand(nheads, dstate, device=device).mul(2))
    B = torch.randn(batch, seqlen, ngroups, dstate, device=device, requires_grad=True)
    C = torch.randn(batch, seqlen, ngroups, dstate, device=device, requires_grad=True)

    # Reference
    out_ref = blelloch_chunk_scan_combined_ref(x, dt, A, B, C, chunk_size=64)
    out_ref.pow(2).sum().backward()
    grads_ref = {}
    for k, var in [('x', x), ('dt', dt), ('B', B), ('C', C)]:
        if var.grad is not None:
            grads_ref[k] = var.grad.clone()
            var.grad = None

    # Blelloch (fresh copies to avoid graph conflicts)
    x2 = x.detach().clone().requires_grad_()
    dt2 = dt.detach().clone().requires_grad_()
    B2 = B.detach().clone().requires_grad_()
    C2 = C.detach().clone().requires_grad_()
    out = blelloch_chunk_scan_combined(x2, dt2, A, B2, C2, chunk_size=64)[0]
    out.pow(2).sum().backward()

    for name, var_ref, var_bl in [('x', x, x2), ('dt', dt, dt2), ('B', B, B2), ('C', C, C2)]:
        if name in grads_ref:
            err = (grads_ref[name] - var_bl.grad).abs().max().item()
            assert err < 1e-3, f"d{name} mismatch: max_err={err:.2e}"


@pytest.mark.parametrize("device", DEVICES)
def test_initial_states(device):
    """Passing initial_states produces correct outputs."""
    torch.manual_seed(42)
    batch, seqlen, nheads, headdim = 1, 8, 2, 4
    dstate, ngroups = 3, 1

    x = torch.randn(batch, seqlen, nheads, headdim, device=device)
    dt = torch.rand(batch, seqlen, nheads, device=device).mul(0.1)
    A = -torch.exp(torch.rand(nheads, dstate, device=device).mul(2))
    B = torch.randn(batch, seqlen, ngroups, dstate, device=device)
    C = torch.randn(batch, seqlen, ngroups, dstate, device=device)
    init = torch.randn(batch, nheads, headdim, dstate, device=device)

    out, final = blelloch_chunk_scan_combined(
        x, dt, A, B, C, chunk_size=64,
        initial_states=init, return_final_states=True,
    )

    ref, ref_final = blelloch_chunk_scan_combined_ref(
        x, dt, A, B, C, chunk_size=64,
        initial_states=init, return_final_states=True,
    )

    err_out = (out - ref).abs().max().item()
    err_final = (final - ref_final).abs().max().item()
    assert err_out < 1e-4, f"initial_states output mismatch: max_err={err_out:.2e}"
    assert err_final < 1e-4, f"initial_states final_state mismatch: max_err={err_final:.2e}"


@pytest.mark.parametrize("device", DEVICES)
def test_dt_limit(device):
    """dt_limit clamps delta values correctly."""
    torch.manual_seed(42)
    batch, seqlen, nheads, headdim = 1, 8, 2, 4
    dstate, ngroups = 3, 1

    x = torch.randn(batch, seqlen, nheads, headdim, device=device)
    dt = torch.rand(batch, seqlen, nheads, device=device).mul(5)
    A = -torch.exp(torch.rand(nheads, dstate, device=device).mul(2))
    B = torch.randn(batch, seqlen, ngroups, dstate, device=device)
    C = torch.randn(batch, seqlen, ngroups, dstate, device=device)

    out_clamped = blelloch_chunk_scan_combined(
        x, dt, A, B, C, chunk_size=64,
        dt_limit=(0.001, 1.0),
    )[0]

    dt_clamped = dt.clamp(min=0.001, max=1.0)
    ref = blelloch_chunk_scan_combined_ref(
        x, dt_clamped, A, B, C, chunk_size=64,
    )

    err = (out_clamped - ref).abs().max().item()
    assert err < 1e-4, f"dt_limit mismatch: {err:.2e}"


@pytest.mark.parametrize("device", DEVICES)
def test_scalar_A(device):
    """Scalar A (nheads,) is expanded to (nheads, dstate)."""
    torch.manual_seed(42)
    batch, seqlen, nheads, headdim = 1, 8, 2, 4
    dstate, ngroups = 3, 1

    x = torch.randn(batch, seqlen, nheads, headdim, device=device)
    dt = torch.rand(batch, seqlen, nheads, device=device).mul(0.1)
    A_scalar = -torch.exp(torch.rand(nheads, device=device).mul(2))
    A_matrix = A_scalar.unsqueeze(-1).expand(-1, dstate)
    B = torch.randn(batch, seqlen, ngroups, dstate, device=device)
    C = torch.randn(batch, seqlen, ngroups, dstate, device=device)

    out_scalar = blelloch_chunk_scan_combined(x, dt, A_scalar, B, C, chunk_size=64)[0]
    out_matrix = blelloch_chunk_scan_combined(x, dt, A_matrix, B, C, chunk_size=64)[0]

    err = (out_scalar - out_matrix).abs().max().item()
    assert err < 1e-6, f"Scalar vs matrix A mismatch: {err:.2e}"


@pytest.mark.parametrize("device", DEVICES)
def test_initial_states_backward(device):
    """Backward passes with initial_states (non-differentiable)."""
    torch.manual_seed(42)
    batch, seqlen, nheads, headdim = 1, 4, 2, 4
    dstate, ngroups = 3, 1

    x = torch.randn(batch, seqlen, nheads, headdim, device=device, requires_grad=True)
    dt = torch.rand(batch, seqlen, nheads, device=device)
    dt.mul_(0.1)
    dt.requires_grad_(True)
    A = -torch.exp(torch.rand(nheads, dstate, device=device).mul(2))
    B = torch.randn(batch, seqlen, ngroups, dstate, device=device, requires_grad=True)
    C = torch.randn(batch, seqlen, ngroups, dstate, device=device, requires_grad=True)
    init = torch.randn(batch, nheads, headdim, dstate, device=device)

    out = blelloch_chunk_scan_combined(
        x, dt, A, B, C, chunk_size=64,
        initial_states=init, return_final_states=True,
    )[0]
    out.pow(2).sum().backward()

    assert x.grad is not None
    assert dt.grad is not None
    assert B.grad is not None
    assert C.grad is not None

    # Compare against reference with initial_states
    x2 = x.detach().clone().requires_grad_()
    dt2 = dt.detach().clone().requires_grad_()
    B2 = B.detach().clone().requires_grad_()
    C2 = C.detach().clone().requires_grad_()

    ref = blelloch_chunk_scan_combined_ref(
        x2, dt2, A, B2, C2, chunk_size=64,
        initial_states=init, return_final_states=True,
    )[0]
    ref.pow(2).sum().backward()

    for name, ref_t, tgt_t in [('x', x2, x), ('dt', dt2, dt), ('B', B2, B), ('C', C2, C)]:
        err = (ref_t.grad - tgt_t.grad).abs().max().item()
        assert err < 1e-3, f"d{name} mismatch with initial_states: {err:.2e}"


@pytest.mark.parametrize("device", DEVICES)
def test_no_z_no_D(device):
    """Works without z or D."""
    torch.manual_seed(42)
    batch, seqlen, nheads, headdim = 1, 8, 2, 4
    dstate, ngroups = 3, 1

    x = torch.randn(batch, seqlen, nheads, headdim, device=device)
    dt = torch.rand(batch, seqlen, nheads, device=device).mul(0.1)
    A = -torch.exp(torch.rand(nheads, dstate, device=device).mul(2))
    B = torch.randn(batch, seqlen, ngroups, dstate, device=device)
    C = torch.randn(batch, seqlen, ngroups, dstate, device=device)

    out = blelloch_chunk_scan_combined(x, dt, A, B, C, chunk_size=64)[0]
    ref = blelloch_chunk_scan_combined_ref(x, dt, A, B, C, chunk_size=64)

    err = (out - ref).abs().max().item()
    assert err < 1e-4, f"no_z_no_D mismatch: {err:.2e}"


@pytest.mark.parametrize("device", DEVICES)
def test_return_final_state(device):
    """return_final_states returns a valid final state."""
    torch.manual_seed(42)
    batch, seqlen, nheads, headdim = 1, 8, 2, 4
    dstate, ngroups = 3, 1

    x = torch.randn(batch, seqlen, nheads, headdim, device=device)
    dt = torch.rand(batch, seqlen, nheads, device=device).mul(0.1)
    A = -torch.exp(torch.rand(nheads, dstate, device=device).mul(2))
    B = torch.randn(batch, seqlen, ngroups, dstate, device=device)
    C = torch.randn(batch, seqlen, ngroups, dstate, device=device)

    out, final = blelloch_chunk_scan_combined(
        x, dt, A, B, C, chunk_size=64,
        return_final_states=True,
    )
    ref, ref_final = blelloch_chunk_scan_combined_ref(
        x, dt, A, B, C, chunk_size=64,
        return_final_states=True,
    )

    assert (out - ref).abs().max().item() < 1e-4, f"Output mismatch: {(out-ref).abs().max().item():.2e}"
    assert (final - ref_final).abs().max().item() < 1e-4, f"Final state mismatch: {(final-ref_final).abs().max().item():.2e}"


if HAS_TRITON:

    @pytest.mark.parametrize("batch", [1, 2])
    @pytest.mark.parametrize("seqlen", [32, 64])
    @pytest.mark.parametrize("nheads", [2, 4])
    @pytest.mark.parametrize("headdim", [8, 16])
    @pytest.mark.parametrize("dstate", [4])
    def test_forward_vs_triton(batch, seqlen, nheads, headdim, dstate):
        """Blelloch forward matches Triton chunk scan on GPU."""
        torch.manual_seed(42)
        ngroups = 1
        device = "cuda"
        chunk_size = min(seqlen, 32)

        x = torch.randn(batch, seqlen, nheads, headdim, device=device)
        dt = F.softplus(torch.randn(batch, seqlen, nheads, device=device) - 2)
        A = -torch.exp(torch.rand(nheads, device=device))
        B = torch.randn(batch, seqlen, ngroups, dstate, device=device)
        C = torch.randn(batch, seqlen, ngroups, dstate, device=device)
        D = torch.randn(nheads, headdim, device=device)
        z = torch.randn(batch, seqlen, nheads, headdim, device=device)
        dt_bias = torch.randn(nheads, device=device)

        out_triton = mamba_chunk_scan_combined(
            x, dt, A, B, C, chunk_size, D=D, z=z,
            dt_bias=dt_bias, dt_softplus=True,
        )

        out_blel = blelloch_chunk_scan_combined(
            x, dt, A, B, C, chunk_size, D=D, z=z,
            dt_bias=dt_bias, dt_softplus=True,
        )[0]

        err = (out_blel - out_triton).abs().max().item()
        rel = err / max(out_triton.abs().max().item(), 1e-30)
        print(f"  Triton vs Blelloch: max_err={err:.2e} rel={rel:.2e}")
        assert rel < 1e-3, f"Triton mismatch: rel={rel:.2e}"
