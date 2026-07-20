import torch
import torch.nn.functional as F
from einops import rearrange, repeat


def blelloch_scan_batched(s_vals, d_vals):
    B, L, D = s_vals.shape
    _, _, _, P = d_vals.shape
    device = s_vals.device

    N = 1
    while N <= L:
        N <<= 1

    if N > L:
        pad_s = torch.ones(B, N - L, D, device=device, dtype=s_vals.dtype)
        pad_d = torch.zeros(B, N - L, D, P, device=device, dtype=d_vals.dtype)
        s_pad = torch.cat([s_vals, pad_s], dim=1)
        d_pad = torch.cat([d_vals, pad_d], dim=1)
    else:
        s_pad = s_vals
        d_pad = d_vals

    st = torch.empty_like(s_pad)
    dt = torch.empty_like(d_pad)

    step = 2
    while step <= N:
        hh = step // 2
        idx_r = torch.arange(step - 1, N, step, device=device)
        idx_l = idx_r - hh

        st[:, idx_r] = s_pad[:, idx_r]
        dt[:, idx_r] = d_pad[:, idx_r]

        s_l = s_pad[:, idx_l]
        d_l = d_pad[:, idx_l]

        s_pad[:, idx_r] = s_l * st[:, idx_r]
        d_pad[:, idx_r] = st[:, idx_r].unsqueeze(-1) * d_l + dt[:, idx_r]
        step <<= 1

    s_pad[:, -1] = 1.0
    d_pad[:, -1] = 0.0

    step_val = N.bit_length() - 2
    while step_val >= 0:
        cur_step = 1 << (step_val + 1)
        hh = cur_step // 2
        idx_r = torch.arange(cur_step - 1, N, cur_step, device=device)
        idx_l = idx_r - hh

        st[:, idx_l] = s_pad[:, idx_l]
        st[:, idx_r] = s_pad[:, idx_r]
        dt[:, idx_l] = d_pad[:, idx_l]
        dt[:, idx_r] = d_pad[:, idx_r]

        s_pad[:, idx_l] = st[:, idx_r]
        d_pad[:, idx_l] = dt[:, idx_r]
        s_pad[:, idx_r] = st[:, idx_r] * st[:, idx_l]
        d_pad[:, idx_r] = st[:, idx_l].unsqueeze(-1) * dt[:, idx_r] + dt[:, idx_l]
        step_val -= 1

    return d_pad[:, 1:L + 1]


def _expand_bc(B_tensor, C_tensor, nheads, dtype=torch.float32):
    B_in = B_tensor.to(dtype)
    C_in = C_tensor.to(dtype)
    if B_in.dim() == 2:
        B_in = B_in.unsqueeze(0).unsqueeze(0)
    if B_in.dim() == 3:
        B_in = B_in.unsqueeze(0)
    if C_in.dim() == 2:
        C_in = C_in.unsqueeze(0).unsqueeze(0)
    if C_in.dim() == 3:
        C_in = C_in.unsqueeze(0)
    nheads_ratio = nheads // B_in.shape[-2]
    if nheads_ratio > 1:
        B_exp = repeat(B_in, "... g d -> ... (g h) d", h=nheads_ratio)
        C_exp = repeat(C_in, "... g d -> ... (g h) d", h=nheads_ratio)
    else:
        B_exp = B_in
        C_exp = C_in
    return B_exp, C_exp


def _reduce_bc(dB, dC, ngroups, nheads, dtype):
    nheads_ratio = nheads // ngroups
    dB_red = torch.zeros(*dB.shape[:-2], ngroups, dB.shape[-1], device=dB.device, dtype=dtype)
    dC_red = torch.zeros(*dC.shape[:-2], ngroups, dC.shape[-1], device=dC.device, dtype=dtype)
    for g in range(ngroups):
        start = g * nheads_ratio
        end = (g + 1) * nheads_ratio
        dB_red[..., g, :] = dB[..., start:end, :].sum(dim=-2)
        dC_red[..., g, :] = dC[..., start:end, :].sum(dim=-2)
    return dB_red, dC_red


def _apply_varlen_mask(y, cu_seqlens, seqlen, orig_batch):
    if cu_seqlens is None:
        return y
    out = torch.zeros(orig_batch, seqlen, *y.shape[2:], device=y.device, dtype=y.dtype)
    for i in range(orig_batch):
        length = min(seqlen, int(cu_seqlens[i+1] - cu_seqlens[i]))
        out[i, :length] = y[i, :length]
    return out


def _complex_to_real(A):
    """Convert complex A to real with doubled dstate (Mamba-1 convention)."""
    A_real = torch.view_as_real(A)
    return A_real.reshape(A.shape[0], -1)


def ssm_fwd(x, dt, A, B, C, D=None, z=None, delta_bias=None,
            delta_softplus=False, return_last_state=False,
            cu_seqlens=None, checkpoint_lvl=0):
    if cu_seqlens is not None:
        orig_batch = len(cu_seqlens) - 1
        batch = orig_batch
        seqlen = x.shape[1]
    else:
        orig_batch = x.shape[0]
        batch = orig_batch
        seqlen = x.shape[1]

    nheads = A.shape[0]
    headdim = x.shape[-1]
    dstate = B.shape[-1]

    orig_dtype = x.dtype
    x_f = x.float()
    dt_f = dt.float()
    if delta_bias is not None:
        dt_f = dt_f + delta_bias.float().unsqueeze(0).unsqueeze(1)
    if delta_softplus:
        dt_f = F.softplus(dt_f)

    is_complex = A.is_complex()
    if is_complex:
        A_f = _complex_to_real(A).float()
    else:
        A_f = A.float()

    if A_f.dim() == 1:
        A_f = A_f.unsqueeze(-1).expand(-1, dstate)

    B_exp, C_exp = _expand_bc(B, C, nheads)

    scale = torch.exp(dt_f.unsqueeze(-1) * A_f.unsqueeze(0).unsqueeze(1))
    drive = dt_f.unsqueeze(-1).unsqueeze(-1) * B_exp.unsqueeze(-1) * x_f.unsqueeze(-2)

    s_bh = rearrange(scale, "b l h d -> (b h) l d")
    d_bh = rearrange(drive, "b l h d p -> (b h) l d p")

    h = blelloch_scan_batched(s_bh, d_bh)
    h = rearrange(h, "(b h) l d p -> b l h d p", b=batch, h=nheads)

    y = torch.einsum("b l h d p, b l h d -> b l h p", h, C_exp)

    if D is not None:
        D_f = D.float()
        if D_f.dim() == 1:
            D_f = D_f.unsqueeze(-1)
        y = y + x_f * D_f.unsqueeze(0).unsqueeze(1)

    if z is not None:
        y = y * F.silu(z.float())

    if cu_seqlens is not None:
        y = _apply_varlen_mask(y, cu_seqlens, seqlen, orig_batch)

    out = y.to(orig_dtype)

    if return_last_state:
        last_state = h[:, -1].permute(0, 2, 3, 1).to(orig_dtype)
        if checkpoint_lvl > 0:
            saved = (x, dt, A, B, C, D, z, delta_bias, scale.detach().float(),
                     drive.detach().float(), B_exp.detach().float(),
                     C_exp.detach().float(), A_f.detach().float(),
                     dt_f.detach().float(), is_complex, orig_dtype)
            return out, last_state, saved
        return out, last_state

    if checkpoint_lvl > 0:
        saved = (x, dt, A, B, C, D, z, delta_bias, scale.detach().float(),
                 drive.detach().float(), B_exp.detach().float(),
                 C_exp.detach().float(), A_f.detach().float(),
                 dt_f.detach().float(), is_complex, orig_dtype)
        return out, saved
    return out


def ssm_bwd(dout, x, dt, A, B, C, D, z, delta_bias,
            delta_softplus=False, scale=None, drive=None,
            B_exp=None, C_exp=None, A_f=None, dt_f=None,
            is_complex=False, orig_dtype=torch.float32, h=None):
    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape

    if dt_f is None or A_f is None or B_exp is None or C_exp is None:
        x_f = x.float().detach()
        dt_f = dt.float().detach()
        if delta_bias is not None:
            dt_f = dt_f + delta_bias.float().unsqueeze(0).unsqueeze(1)
        if delta_softplus:
            dt_f = F.softplus(dt_f)
        if is_complex:
            A_f = _complex_to_real(A).float().detach()
        else:
            A_f = A.float().detach()
        if A_f.dim() == 1:
            A_f = A_f.unsqueeze(-1).expand(-1, dstate)
        B_exp, C_exp = _expand_bc(B, C, nheads)
        scale = torch.exp(dt_f.unsqueeze(-1) * A_f.unsqueeze(0).unsqueeze(1))
        drive = dt_f.unsqueeze(-1).unsqueeze(-1) * B_exp.unsqueeze(-1) * x_f.unsqueeze(-2)
    else:
        x_f = x.float().detach()
        dt_f = dt_f.detach()
        A_f = A_f.detach()
        B_exp = B_exp.detach()
        C_exp = C_exp.detach()
        scale = scale.detach()
        drive = drive.detach()

    s_exp = scale.unsqueeze(-1)

    if z is not None:
        z_f = z.float().detach()
        if z_f.dim() == 3:
            z_f = z_f.unsqueeze(-1)
        dout_f = dout.float() * F.silu(z_f)
    else:
        z_f = None
        dout_f = dout.float()

    if h is not None:
        h = h.float().detach()
    else:
        s_bh = rearrange(scale, "b l h d -> (b h) l d")
        d_bh = rearrange(drive, "b l h d p -> (b h) l d p")
        h = blelloch_scan_batched(s_bh, d_bh)
        h = rearrange(h, "(b h) l d p -> b l h d p", b=batch, h=nheads)

    h_adj = torch.zeros(batch, nheads, dstate, headdim, device=x.device, dtype=torch.float32)
    h_adj_list = []
    for t in reversed(range(seqlen)):
        dout_y = torch.einsum("b h p, b h d -> b h d p", dout_f[:, t], C_exp[:, t])
        if t < seqlen - 1:
            h_adj = dout_y + s_exp[:, t + 1] * h_adj
        else:
            h_adj = dout_y
        h_adj_list.append(h_adj.unsqueeze(1))
    h_adj_list.reverse()
    h_adj_stacked = torch.cat(h_adj_list, dim=1)

    h_prev = torch.cat([
        torch.zeros(batch, 1, nheads, dstate, headdim, device=x.device, dtype=torch.float32),
        h[:, :-1]
    ], dim=1)

    dscale = torch.einsum("b l h d p, b l h d p -> b l h d", h_adj_stacked, h_prev)
    ddrive = h_adj_stacked

    dA_raw = torch.einsum("b l h d, b l h d -> h d", dscale, dt_f.unsqueeze(-1) * scale)

    ddt = torch.einsum("b l h d, h d, b l h d -> b l h", dscale, A_f, scale)
    ddt = ddt + torch.einsum("b l h d p, b l h d, b l h p -> b l h", ddrive, B_exp, x_f)

    dB = torch.einsum("b l h d p, b l h p -> b l h d", ddrive, dt_f.unsqueeze(-1) * x_f)
    dx = torch.einsum("b l h d p, b l h d -> b l h p", ddrive, dt_f.unsqueeze(-1) * B_exp)
    dC = torch.einsum("b l h p, b l h d p -> b l h d", dout_f, h)

    if D is not None:
        D_f = D.float().detach()
        if D_f.dim() == 1:
            D_f = D_f.unsqueeze(-1)
        dD = (dout_f * x_f).sum(dim=(0, 1))
        dx = dx + dout_f * D_f.unsqueeze(0).unsqueeze(1)
    else:
        D_f = None
        dD = None

    if z is not None:
        y_ssm = torch.einsum("b l h d p, b l h d -> b l h p", h, C_exp)
        if D is not None:
            y_ssm = y_ssm + x_f * D_f.unsqueeze(0).unsqueeze(1)
        silu_deriv = torch.sigmoid(z_f) * (1 + z_f * (1 - torch.sigmoid(z_f)))
        dz = dout.float() * y_ssm * silu_deriv
        if dz.dim() == 4 and z.dim() == 3:
            dz = dz.squeeze(-1)
    else:
        dz = None

    if is_complex:
        dA_complex = torch.view_as_complex(dA_raw.float().reshape(nheads, -1, 2).contiguous())
        dA_out = dA_complex.to(A.dtype)
    else:
        if A.shape[-1] != dstate:
            dA_out = dA_raw.sum(-1, keepdim=True).to(A.dtype)
        else:
            dA_out = dA_raw.to(A.dtype)

    dB_red, dC_red = _reduce_bc(dB, dC, ngroups, nheads, torch.float32)

    if delta_softplus:
        sp_in = dt.float() + (delta_bias.float().unsqueeze(0).unsqueeze(1) if delta_bias is not None else 0)
        sp_deriv = torch.sigmoid(sp_in)
        ddt = ddt * sp_deriv
        ddelta_bias = ddt.sum(dim=(0, 1)) if delta_bias is not None else None
    else:
        ddelta_bias = ddt.sum(dim=(0, 1)) if delta_bias is not None else None

    return (dx.to(x.dtype), ddt.to(dt.dtype), dA_out,
            dB_red.to(B.dtype), dC_red.to(C.dtype),
            dD.to(D.dtype) if dD is not None else None,
            dz.to(z.dtype) if dz is not None else None,
            ddelta_bias, None, None, None, None)


class BlellochSSMFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, dt, A, B, C, D, z, delta_bias, delta_softplus, return_last_state, cu_seqlens, checkpoint_lvl):
        batch, seqlen, nheads, headdim = x.shape
        _, _, ngroups, dstate = B.shape

        orig_dtype = x.dtype
        x_f = x.float()
        dt_f = dt.float()
        if delta_bias is not None:
            dt_f = dt_f + delta_bias.float().unsqueeze(0).unsqueeze(1)
        if delta_softplus:
            dt_f = F.softplus(dt_f)

        is_complex = A.is_complex()
        if is_complex:
            A_f = _complex_to_real(A).float()
        else:
            A_f = A.float()
        if A_f.dim() == 1:
            A_f = A_f.unsqueeze(-1).expand(-1, dstate)

        B_exp, C_exp = _expand_bc(B, C, nheads)

        scale = torch.exp(dt_f.unsqueeze(-1) * A_f.unsqueeze(0).unsqueeze(1))
        drive = dt_f.unsqueeze(-1).unsqueeze(-1) * B_exp.unsqueeze(-1) * x_f.unsqueeze(-2)

        s_bh = rearrange(scale, "b l h d -> (b h) l d")
        d_bh = rearrange(drive, "b l h d p -> (b h) l d p")

        h = blelloch_scan_batched(s_bh, d_bh)
        h = rearrange(h, "(b h) l d p -> b l h d p", b=batch, h=nheads)

        y = torch.einsum("b l h d p, b l h d -> b l h p", h, C_exp)

        if D is not None:
            D_f = D.float()
            if D_f.dim() == 1:
                D_f = D_f.unsqueeze(-1)
            y = y + x_f * D_f.unsqueeze(0).unsqueeze(1)
        if z is not None:
            z_f = z.float()
            if z_f.dim() == 3:
                z_f = z_f.unsqueeze(-1)
            y = y * F.silu(z_f)

        if cu_seqlens is not None:
            y = _apply_varlen_mask(y, cu_seqlens, seqlen, len(cu_seqlens)-1)

        out = y.to(orig_dtype)

        ctx.delta_softplus = delta_softplus
        ctx.has_D = D is not None
        ctx.has_z = z is not None
        ctx.has_delta_bias = delta_bias is not None
        ctx.is_complex = is_complex
        ctx.checkpoint_lvl = checkpoint_lvl
        ctx.orig_dtype = orig_dtype

        if checkpoint_lvl == 0:
            ctx.save_for_backward(x, dt, A, B, C, h.detach().float(),
                                  torch.tensor(0, dtype=torch.int) if D is None else D,
                                  torch.tensor(0, dtype=torch.int) if z is None else z,
                                  torch.tensor(0, dtype=torch.int) if delta_bias is None else delta_bias)
        else:
            ctx.save_for_backward(x, dt, A, B, C,
                                  torch.tensor(0, dtype=torch.int) if D is None else D,
                                  torch.tensor(0, dtype=torch.int) if z is None else z,
                                  torch.tensor(0, dtype=torch.int) if delta_bias is None else delta_bias,
                                  scale.detach().float(),
                                  drive.detach().float(),
                                  B_exp.detach().float(),
                                  C_exp.detach().float(),
                                  A_f.detach().float(),
                                  dt_f.detach().float())

        if return_last_state:
            last_state = h[:, -1].permute(0, 2, 3, 1).to(orig_dtype)
            ctx.mark_non_differentiable(last_state)
            return out, last_state
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        x, dt, A, B, C = ctx.saved_tensors[:5]

        if ctx.checkpoint_lvl > 0:
            D = ctx.saved_tensors[5] if ctx.has_D else None
            z = ctx.saved_tensors[6] if ctx.has_z else None
            delta_bias = ctx.saved_tensors[7] if ctx.has_delta_bias else None
            scale, drive, B_exp, C_exp, A_f, dt_f = ctx.saved_tensors[8:14]
            h = None
        else:
            h = ctx.saved_tensors[5]
            D = ctx.saved_tensors[6] if ctx.has_D else None
            z = ctx.saved_tensors[7] if ctx.has_z else None
            delta_bias = ctx.saved_tensors[8] if ctx.has_delta_bias else None
            scale = drive = B_exp = C_exp = A_f = dt_f = None

        return ssm_bwd(dout, x, dt, A, B, C, D, z, delta_bias,
                       ctx.delta_softplus,
                       scale=scale, drive=drive,
                       B_exp=B_exp, C_exp=C_exp,
                       A_f=A_f, dt_f=dt_f,
                       is_complex=getattr(ctx, 'is_complex', False),
                       orig_dtype=getattr(ctx, 'orig_dtype', torch.float32),
                       h=h)


def blelloch_ssm(x, dt, A, B, C, D=None, z=None, delta_bias=None,
                 delta_softplus=False, return_last_state=False,
                 cu_seqlens=None, checkpoint_lvl=0):
    return BlellochSSMFn.apply(x, dt, A, B, C, D, z, delta_bias, delta_softplus,
                                return_last_state, cu_seqlens, checkpoint_lvl)


def ref_ssm_scan(x, dt, A, B, C, D=None, z=None, delta_bias=None,
                 delta_softplus=False, return_last_state=False):
    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape
    nheads_ratio = nheads // ngroups

    B_exp = repeat(B.float(), "b l g d -> b l (g h) d", h=nheads_ratio)
    C_exp = repeat(C.float(), "b l g d -> b l (g h) d", h=nheads_ratio)

    dt_f = dt.float()
    if delta_bias is not None:
        dt_f = dt_f + delta_bias.float().unsqueeze(0).unsqueeze(1)
    if delta_softplus:
        dt_f = F.softplus(dt_f)

    A_f = A.float()
    scale = torch.exp(dt_f.unsqueeze(-1) * A_f.unsqueeze(0).unsqueeze(1))

    h = torch.zeros(batch, nheads, dstate, headdim, device=x.device, dtype=torch.float32)
    ys = []
    for t in range(seqlen):
        drive = (dt_f[:, t, :].unsqueeze(-1).unsqueeze(-1) *
                 B_exp[:, t].unsqueeze(-1) *
                 x[:, t].float().unsqueeze(-2))
        h = scale[:, t].unsqueeze(-1) * h + drive
        y = torch.einsum("b h d p, b h d -> b h p", h, C_exp[:, t])
        ys.append(y)

    y = torch.stack(ys, dim=1)
    if D is not None:
        y = y + x.float() * D.float().unsqueeze(0).unsqueeze(1)
    if z is not None:
        z_f = z.float()
        if z_f.dim() == 3:
            z_f = z_f.unsqueeze(-1)
        y = y * F.silu(z_f)
    out = y.to(x.dtype)

    if return_last_state:
        return out, h.clone().permute(0, 2, 3, 1).mul(1).to(x.dtype)
    return out


def test_blelloch():
    torch.manual_seed(42)

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
                            ref_max = ref.abs().max().item()
                            rel_err = err / max(ref_max, 1e-30)

                            if rel_err > 1e-4:
                                print(f"FAIL forward: b={batch} L={seqlen} h={nheads} g={ngroups} ds={dstate} hd={headdim}: rel_err={rel_err:.2e}")
                                return False

    print(f"PASS: All forward tests passed (max rel_err < 1e-4)")

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

    all_pass = True
    for name in ['x', 'dt', 'A', 'B', 'C', 'D', 'z', 'delta_bias']:
        ref_grad = grads_ref[name]
        grad = locals()[name].grad
        err = (ref_grad - grad).abs().max().item()
        ref_max = ref_grad.abs().max().item()
        rel_err = err / max(ref_max, 1e-30)
        if rel_err >= 1e-3:
            print(f"FAIL backward d{name}: rel_err={rel_err:.6e}")
            all_pass = False

    if all_pass:
        print(f"PASS: All backward tests passed (max rel_err < 1e-3)")
    return all_pass


if __name__ == "__main__":
    test_blelloch()
