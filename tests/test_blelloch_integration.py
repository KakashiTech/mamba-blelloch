"""Comprehensive tests for the Blelloch prefix scan SSM implementation."""

import torch
import torch.nn.functional as F
from einops import repeat
import sys

sys.path.insert(0, '/tmp/opencode/mamba-forensic/mamba_ssm/ops')

from einops import rearrange, repeat

from selective_scan_blelloch import (
    BlellochSSMFn, ssm_fwd, ssm_bwd, ref_ssm_scan, blelloch_scan_batched,
    blelloch_ssm
)


def test_forward_small():
    """Forward against ref_ssm_scan for diverse configurations."""
    torch.manual_seed(42)
    failures = []
    for batch in [1, 2]:
        for seqlen in [4, 8, 16, 32, 64, 128]:
            for nheads in [1, 2, 4]:
                for ngroups in [1, 2]:
                    if nheads % ngroups != 0:
                        continue
                    for dstate in [2, 4, 8, 16, 32]:
                        for headdim in [2, 4, 8, 16, 32]:
                            if dstate * headdim > 512:
                                continue
                            x = torch.randn(batch, seqlen, nheads, headdim)
                            dt = torch.rand(batch, seqlen, nheads).mul(0.1)
                            A = -torch.exp(torch.rand(nheads, dstate).mul(2))
                            B = torch.randn(batch, seqlen, ngroups, dstate)
                            C = torch.randn(batch, seqlen, ngroups, dstate)
                            ref = ref_ssm_scan(x, dt, A, B, C)
                            out = ssm_fwd(x, dt, A, B, C)
                            err = (ref.double() - out.double()).abs().max().item()
                            rel = err / max(ref.abs().max().item(), 1e-30)
                            if rel > 1e-4:
                                failures.append(
                                    f"b={batch} L={seqlen} h={nheads} "
                                    f"g={ngroups} d={dstate} p={headdim} rel={rel:.2e}")
    assert not failures, f"Forward failures: {len(failures)}"


def test_backward_gradients():
    """Compare all 8 gradients against reference autograd."""
    torch.manual_seed(42)
    batch, seqlen, nheads, headdim = 2, 8, 2, 4
    dstate, ngroups = 3, 1

    A = torch.nn.Parameter(-torch.exp(torch.rand(nheads, dstate).mul(2)))
    x = torch.randn(batch, seqlen, nheads, headdim, requires_grad=True)
    dt = torch.rand(batch, seqlen, nheads, requires_grad=True)
    B = torch.randn(batch, seqlen, ngroups, dstate, requires_grad=True)
    C = torch.randn(batch, seqlen, ngroups, dstate, requires_grad=True)
    D = torch.randn(nheads, headdim, requires_grad=True)
    z = torch.randn(batch, seqlen, nheads, headdim, requires_grad=True)
    delta_bias = torch.randn(nheads, requires_grad=True)

    ref_out = ref_ssm_scan(x, dt, A, B, C, D, z, delta_bias, delta_softplus=True)
    ref_loss = ref_out.pow(2).sum()
    ref_loss.backward()
    grads_ref = {}
    for k in ['x', 'dt', 'A', 'B', 'C', 'D', 'z', 'delta_bias']:
        grads_ref[k] = locals()[k].grad.clone()
        locals()[k].grad.zero_()

    blel_out = BlellochSSMFn.apply(x, dt, A, B, C, D, z, delta_bias, True)
    blel_loss = blel_out.pow(2).sum()
    blel_loss.backward()

    failures = []
    for name in ['x', 'dt', 'A', 'B', 'C', 'D', 'z', 'delta_bias']:
        err = (grads_ref[name] - locals()[name].grad).abs().max().item()
        ref_max = grads_ref[name].abs().max().item()
        rel = err / max(ref_max, 1e-30)
        if rel >= 1e-3:
            failures.append(f"d{name}: rel={rel:.6e}")
    assert not failures, f"Gradient failures: {failures}"


def test_constant_bc():
    """Test constant B/C in (dim, dstate) format."""
    torch.manual_seed(42)
    batch, seqlen, nheads, headdim = 2, 8, 2, 4
    dstate, ngroups = 3, 1

    x = torch.randn(batch, seqlen, nheads, headdim)
    dt = torch.rand(batch, seqlen, nheads).mul(0.1)
    A = -torch.exp(torch.rand(nheads, dstate).mul(2))

    B_var = torch.randn(batch, seqlen, ngroups, dstate)
    C_var = torch.randn(batch, seqlen, ngroups, dstate)
    B_const = torch.randn(ngroups, dstate)
    C_const = torch.randn(ngroups, dstate)

    ref_var = ref_ssm_scan(x, dt, A, B_var, C_var)
    ref_const = ref_ssm_scan(x, dt, A,
                              B_const.unsqueeze(0).unsqueeze(0).expand(batch, seqlen, -1, -1),
                              C_const.unsqueeze(0).unsqueeze(0).expand(batch, seqlen, -1, -1))
    out_var = ssm_fwd(x, dt, A, B_var, C_var)
    out_const = ssm_fwd(x, dt, A, B_const, C_const)

    err_var = (ref_var - out_var).abs().max().item()
    err_const = (ref_const - out_const).abs().max().item()
    assert err_var < 1e-4, f"Variable B/C err={err_var:.2e}"
    assert err_const < 1e-4, f"Const B/C err={err_const:.2e}"


def test_checkpoint_lvl0():
    """Checkpoint lvl=0 (default, saves h)."""
    torch.manual_seed(42)
    batch, seqlen, nheads, headdim = 1, 8, 2, 4
    dstate, ngroups = 3, 1

    A = torch.nn.Parameter(-torch.exp(torch.rand(nheads, dstate).mul(2)))
    x = torch.randn(batch, seqlen, nheads, headdim, requires_grad=True)
    dt = torch.rand(batch, seqlen, nheads, requires_grad=True)
    B = torch.randn(batch, seqlen, ngroups, dstate, requires_grad=True)
    C = torch.randn(batch, seqlen, ngroups, dstate, requires_grad=True)

    ref_out = ref_ssm_scan(x, dt, A, B, C)
    ref_out.pow(2).sum().backward()
    grads = {k: locals()[k].grad.clone() for k in ['x', 'dt', 'A', 'B', 'C']}
    for v in [x, dt, A, B, C]:
        v.grad.zero_()

    out = blelloch_ssm(x, dt, A, B, C, checkpoint_lvl=0)
    out.pow(2).sum().backward()

    for name in ['x', 'dt', 'A', 'B', 'C']:
        err = (grads[name] - locals()[name].grad).abs().max().item()
        assert err < 1e-4, f"checkpoint_lvl=0 d{name} err={err:.2e}"


def test_checkpoint_lvl1():
    """Checkpoint lvl=1 (recompute h in backward)."""
    torch.manual_seed(42)
    batch, seqlen, nheads, headdim = 1, 8, 2, 4
    dstate, ngroups = 3, 1

    A = torch.nn.Parameter(-torch.exp(torch.rand(nheads, dstate).mul(2)))
    x = torch.randn(batch, seqlen, nheads, headdim, requires_grad=True)
    dt = torch.rand(batch, seqlen, nheads, requires_grad=True)
    B = torch.randn(batch, seqlen, ngroups, dstate, requires_grad=True)
    C = torch.randn(batch, seqlen, ngroups, dstate, requires_grad=True)

    ref_out = ref_ssm_scan(x, dt, A, B, C)
    ref_out.pow(2).sum().backward()
    grads = {k: locals()[k].grad.clone() for k in ['x', 'dt', 'A', 'B', 'C']}
    for v in [x, dt, A, B, C]:
        v.grad.zero_()

    out = blelloch_ssm(x, dt, A, B, C, checkpoint_lvl=1)
    out.pow(2).sum().backward()

    for name in ['x', 'dt', 'A', 'B', 'C']:
        err = (grads[name] - locals()[name].grad).abs().max().item()
        assert err < 1e-4, f"checkpoint_lvl=1 d{name} err={err:.2e}"


def test_return_last_state():
    """Verify return_last_state matches reference."""
    torch.manual_seed(42)
    batch, seqlen, nheads, headdim = 1, 8, 2, 4
    dstate, ngroups = 3, 1

    A = torch.nn.Parameter(-torch.exp(torch.rand(nheads, dstate).mul(2)))
    x = torch.randn(batch, seqlen, nheads, headdim)
    dt = torch.rand(batch, seqlen, nheads)
    B = torch.randn(batch, seqlen, ngroups, dstate)
    C = torch.randn(batch, seqlen, ngroups, dstate)

    ref_out, ref_state = ref_ssm_scan(x, dt, A, B, C, return_last_state=True)
    out, state = BlellochSSMFn.apply(x, dt, A, B, C, None, None, None, False, True)

    err_out = (ref_out - out).abs().max().item()
    err_state = (ref_state - state).abs().max().item()
    assert err_out < 1e-4, f"last_state output err={err_out:.2e}"
    assert err_state < 1e-4, f"last_state err={err_state:.2e}"


def test_interface_selective_scan_fn():
    """Test that blelloch_interface matches ref when converted properly."""
    torch.manual_seed(42)
    from selective_scan_blelloch_interface import selective_scan_fn as blel_fn

    batch, dim, seqlen = 2, 4, 8
    dstate = 3

    u = torch.randn(batch, dim, seqlen)
    delta = torch.rand(batch, dim, seqlen).mul(0.1)
    A = -torch.exp(torch.rand(dim, dstate).mul(2))
    B = torch.randn(batch, dstate, seqlen)
    C = torch.randn(batch, dstate, seqlen)
    D = torch.randn(dim)
    z = torch.randn(batch, dim, seqlen)
    delta_bias = torch.randn(dim)

    out_ref = ref_ssm_scan(
        rearrange(u, "b d l -> b l d").unsqueeze(-1),
        rearrange(delta, "b d l -> b l d"),
        A,
        repeat(B, "b n l -> b l 1 n"),
        repeat(C, "b n l -> b l 1 n"),
        D.view(-1, 1),
        rearrange(z, "b d l -> b l d"),
        delta_bias,
        delta_softplus=True,
    )
    out_ref = rearrange(out_ref.squeeze(-1), "b l d -> b d l")

    out = blel_fn(u, delta, A, B, C, D, z, delta_bias, delta_softplus=True)

    err = (out_ref - out).abs().max().item()
    assert err < 1e-4, f"Interface mismatch err={err:.2e}"


def test_dtype_preservation():
    """Test that input dtype is preserved in output."""
    torch.manual_seed(42)
    batch, seqlen, nheads, headdim = 1, 8, 2, 4
    dstate, ngroups = 3, 1

    A = -torch.exp(torch.rand(nheads, dstate).mul(2))
    B = torch.randn(batch, seqlen, ngroups, dstate)
    C = torch.randn(batch, seqlen, ngroups, dstate)

    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        x = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype)
        dt = torch.rand(batch, seqlen, nheads, dtype=dtype).mul(0.1)
        a = A.to(dtype)
        b = B.to(dtype)
        c = C.to(dtype)

        out = BlellochSSMFn.apply(x, dt, a, b, c)
        assert out.dtype == dtype, f"dtype {dtype}: output is {out.dtype}"


def test_batched_scan_vs_sequential():
    """Verify batched Blelloch scan matches sequential for each head."""
    torch.manual_seed(42)
    B, L, D, P = 3, 8, 4, 5
    s = torch.randn(B, L, D)
    d = torch.randn(B, L, D, P)

    h_batched = blelloch_scan_batched(s, d)

    h_seq = []
    for b in range(B):
        h_prev = torch.zeros(D, P)
        hs = []
        for t in range(L):
            h_prev = s[b, t].unsqueeze(-1) * h_prev + d[b, t]
            hs.append(h_prev.clone())
        h_seq.append(torch.stack(hs, dim=0))
    h_seq = torch.stack(h_seq, dim=0)

    err = (h_batched - h_seq).abs().max().item()
    assert err < 1e-5, f"Batched scan mismatch err={err:.2e}"


def test_large_seqlen():
    """Test seqlen=2048 (one config)."""
    torch.manual_seed(42)
    batch, seqlen, nheads, headdim = 1, 2048, 1, 2
    dstate, ngroups = 2, 1

    x = torch.randn(batch, seqlen, nheads, headdim)
    dt = torch.rand(batch, seqlen, nheads).mul(0.1)
    A = -torch.exp(torch.rand(nheads, dstate).mul(2))
    B = torch.randn(batch, seqlen, ngroups, dstate)
    C = torch.randn(batch, seqlen, ngroups, dstate)

    ref = ref_ssm_scan(x, dt, A, B, C)
    out = ssm_fwd(x, dt, A, B, C)

    err = (ref.double() - out.double()).abs().max().item()
    rel = err / max(ref.abs().max().item(), 1e-30)
    assert rel < 1e-4, f"Large seqlen err={rel:.2e}"


def test_complex_a():
    """Complex A is realified (dstate doubled), forward matches ref."""
    torch.manual_seed(42)
    batch, seqlen, nheads, headdim = 1, 8, 2, 2
    ngroups = 1
    dstate_c = 2

    x = torch.randn(batch, seqlen, nheads, headdim)
    dt = torch.rand(batch, seqlen, nheads).mul(0.1)
    A_c = torch.complex(-torch.exp(torch.rand(nheads, dstate_c).mul(2)),
                        torch.randn(nheads, dstate_c).mul(0.5))
    B = torch.randn(batch, seqlen, ngroups, dstate_c * 2)
    C = torch.randn(batch, seqlen, ngroups, dstate_c * 2)

    from selective_scan_blelloch import _complex_to_real
    A_real = _complex_to_real(A_c)
    ref = ref_ssm_scan(x, dt, A_real, B, C)
    out = ssm_fwd(x, dt, A_c, B, C)

    err = (ref.double() - out.double()).abs().max().item()
    rel = err / max(ref.abs().max().item(), 1e-30)
    assert rel < 1e-4, f"Complex A err={rel:.2e}"


def test_complex_a_grad():
    """Backward through complex A produces valid gradients."""
    torch.manual_seed(42)
    batch, seqlen, nheads, headdim = 1, 8, 2, 2
    ngroups = 1
    dstate_c = 2

    A_c = torch.nn.Parameter(torch.complex(
        -torch.exp(torch.rand(nheads, dstate_c).mul(2)),
        torch.randn(nheads, dstate_c).mul(0.5)))
    x = torch.randn(batch, seqlen, nheads, headdim, requires_grad=True)
    dt = torch.rand(batch, seqlen, nheads, requires_grad=True)
    B = torch.randn(batch, seqlen, ngroups, dstate_c * 2, requires_grad=True)
    C = torch.randn(batch, seqlen, ngroups, dstate_c * 2, requires_grad=True)

    from selective_scan_blelloch import BlellochSSMFn
    out = BlellochSSMFn.apply(x, dt, A_c, B, C)
    out.pow(2).sum().backward()

    assert A_c.grad is not None
    assert not torch.isnan(A_c.grad.abs()).any(), "Complex A grad has NaN"
    assert A_c.grad.shape == A_c.shape
    print(f"  Complex A grad ok — max |grad| = {A_c.grad.abs().max().item():.4f}")


def test_varlen():
    """Varlen (cu_seqlens) — masked output matches per-sequence ref."""
    torch.manual_seed(42)
    batch, seqlen, nheads, headdim = 2, 8, 2, 4
    dstate, ngroups = 3, 1

    x = torch.randn(batch, seqlen, nheads, headdim)
    dt = torch.rand(batch, seqlen, nheads).mul(0.1)
    A = -torch.exp(torch.rand(nheads, dstate).mul(2))
    B = torch.randn(batch, seqlen, ngroups, dstate)
    C = torch.randn(batch, seqlen, ngroups, dstate)

    ref_full = ref_ssm_scan(x, dt, A, B, C)

    cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.long)
    out = ssm_fwd(x, dt, A, B, C, cu_seqlens=cu_seqlens)

    err = (ref_full[:1, :4] - out[:1, :4]).abs().max().item()
    assert err < 1e-4, f"Varlen first seq err={err:.2e}"
