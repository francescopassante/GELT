"""Quick diagnostic for the 2×2 Wilson-loop stall on GELT.

Goals
-----
1. Confirm that loss=1 = Var(y_norm) ⇒ the model is collapsing to a constant.
2. Quantify how much of the GELT output, at init and during training, is a
   site-local function of P(x) (call it L) vs. a genuine multi-site coupling
   (call it M = total - L).  If only L is growing, the model is doomed:
   for the 2×2 Wilson loop, E[f(P(x)) · y(x)] = 0 ⇒ any site-local predictor
   has MSE ≥ Var(y).
3. Watch α and ‖fc2‖ to see whether the ReZero cascade ever unfreezes.

This is intentionally small (small lattice, few epochs) — we just need to
see the trend.
"""
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim

from gelt import Z2, build_plaquette_datasets, haar_ensemble
from gelt.blocks import GELT
from gelt.lattice import rectangular_wilson_loop


def site_local_proj(model_outputs: torch.Tensor, plaquettes: torch.Tensor) -> tuple:
    """Per-site linear projection of model output onto plaquettes at x.

    Fits ŷ_local(x) = a + Σ_c β_c P_c(x) by least squares using ALL sites/configs
    pooled. Returns (var_local, var_total, var_multi_site).

    For Haar Z₂, P_c(x) ∈ {±1}, mean 0, variance 1, pairwise independent across
    c and across non-overlapping sites. So this projection captures exactly the
    site-local linear component.
    """
    # outputs : (B, *Λ),  plaquettes : (B, C, *Λ, 1, 1)
    y = model_outputs.flatten()  # (B·|Λ|,)
    P = plaquettes.squeeze(-1).squeeze(-1)  # (B, C, *Λ)
    # rearrange to (B·|Λ|, C)
    C = P.shape[1]
    P = P.movedim(1, -1).reshape(-1, C).real.float()
    y = y.float()
    # least squares: y ≈ X β with X = [1, P]
    X = torch.cat([torch.ones(P.shape[0], 1), P], dim=1)
    beta, *_ = torch.linalg.lstsq(X, y.unsqueeze(-1))
    y_hat = (X @ beta).squeeze(-1)
    var_total = y.var(unbiased=False).item()
    var_local = y_hat.var(unbiased=False).item()
    var_resid = (y - y_hat).var(unbiased=False).item()
    return var_local, var_total, var_resid, beta.squeeze(-1).tolist()


def main():
    torch.manual_seed(0)
    D, L = 3, 6  # smaller lattice for speed
    R = 2
    gg = Z2()
    loop_R, loop_T, mu, nu = 2, 2, 0, 1

    print(f"=== Building dataset (D={D}, L={L}, 2×2 Wilson loop, μν={mu}{nu}) ===")
    train, val, test = build_plaquette_datasets(
        N=400, D=D, L=L, gaugegroup=gg, R=R,
        splits=[0.7, 0.15, 0.15], save=False,
        structured=True, sampler=haar_ensemble, beta=1.0,
        target=partial(rectangular_wilson_loop, R=loop_R, T=loop_T, mu=mu, nu=nu),
        n_therm=200, n_skip=5, dtype=torch.float32,
    )

    # standardize y on train split
    y_train = train.dataset.tensors[-1][train.indices]
    mu_y = y_train.mean()
    sigma_y = y_train.std(unbiased=False).clamp_min(1e-12)
    print(f"target raw stats:  μ_y={mu_y.item():.4f}   σ_y={sigma_y.item():.4f}")
    train.dataset.tensors[-1].sub_(mu_y).div_(sigma_y)

    # ---- direct target / input correlation sanity check ---------------------
    Xall = train.dataset.tensors[0]                # (N, C, *Λ, 1, 1)
    yall = train.dataset.tensors[-1]               # (N, *Λ)
    v_loc, v_tot, v_res, betas = site_local_proj(yall, Xall)
    print(
        f"  Per-site linear fit of TARGET onto P(x):"
        f"  var(y) = {v_tot:.4f}  var(ŷ_local) = {v_loc:.6f}  var(resid) = {v_res:.4f}"
    )
    print(f"  least-squares betas = {[f'{b:.4f}' for b in betas]}")
    print("  ⇒ if var(ŷ_local) ≈ 0, the target is orthogonal to {1, P(x)} at every site,")
    print("    so any site-local predictor has MSE ≥ var(y).  This is the trap.")

    # ---- build model --------------------------------------------------------
    model = GELT(
        gaugegroup=gg, L=L, D=D, R=R,
        nhead=4, gemhsa_layers=3, d_qkv=16,
        gate="softplus", dtype=torch.complex64,
        mlp_hidden=32, mlp_out=1,
        reduction="none",
        alpha_init=0.5, init_scale=10.0, mlp_zero_init=False,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"GELT n_params = {n_params}")

    # ---- characterize OUTPUT at init: what fraction is site-local? ----------
    model.eval()
    with torch.no_grad():
        Xb = Xall[:64]
        Tb = train.dataset.tensors[1][:64]
        outb = model(Xb, Tb)
    v_loc, v_tot, v_res, betas = site_local_proj(outb, Xb)
    print(
        f"  INIT model output stats:"
        f"  var(out)={v_tot:.6f}  var(out_local)={v_loc:.6f}  var(out_multi)={v_res:.6f}"
        f"   ratio multi/total = {v_res / max(v_tot, 1e-12):.3f}"
    )

    # ---- training loop with diagnostics -------------------------------------
    train_loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size=64, shuffle=False)
    crit = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=1e-2)

    for epoch in range(20):
        model.train()
        train_loss = 0.0
        n = 0
        for X, T, y in train_loader:
            opt.zero_grad()
            out = model(X, T)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            train_loss += loss.item() * y.shape[0]
            n += y.shape[0]
        train_loss /= n

        model.eval()
        with torch.no_grad():
            v_loss = 0.0; vn = 0
            for X, T, y in val_loader:
                out = model(X, T)
                v_loss += crit(out, y).item() * y.shape[0]
                vn += y.shape[0]
            v_loss /= vn

            Xb = Xall[:64]; Tb = train.dataset.tensors[1][:64]
            outb = model(Xb, Tb)
            v_loc, v_tot, v_res, betas = site_local_proj(outb, Xb)

        alphas = [f"{layer.alpha.item():+.3f}" for layer in model.gemhsa_models]
        fc2 = model.mlp.fc2.weight.detach().abs().mean().item()
        print(
            f"ep {epoch+1:>2d} | train {train_loss:.4f} | val {v_loss:.4f} |"
            f" α=[{','.join(alphas)}] |fc2|̄={fc2:.4f} |"
            f" var(out)={v_tot:.4f} loc/tot={v_loc/max(v_tot,1e-12):.2f}"
        )


if __name__ == "__main__":
    main()
