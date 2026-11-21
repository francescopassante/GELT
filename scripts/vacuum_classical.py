"""Force-matching least squares for every classical vacuum ansatz, and its ε.

Φ_arm(U) = Σ_k c_k O_k(U) is linear in c, so force matching is ordinary least
squares on the Lie gradients: with g_k = ∇O_k per slice and F the temporal
force, c = A⁻¹b, A_kl = Σ g_k·g_l, b_k = Σ g_k·F, summed over every slice of
VAC_FIT_CONFIGS training configurations. One pass computes A and b for the
union basis (vacuum_common.UNION_LABELS); each arm is a sub-block. Complex128.

The arms (vacuum_common.ARMS): `dr` the rescaled-coupling form ψ₀² ∝ e^{−β' S_s}
that dimensional reduction predicts at long wavelength; `shapes` thin loops;
`shapes_sm` the five shapes at APE 0,2,4,6 (the 0⁺⁺ study's strongest
single-scale span); `bb` a Greensite–Olejník-type B·K·B with a free radial
kernel; `dr_bb`; `full` everything (24 + BB_R operators).

Reported per arm: the training and validation force-matching loss per
component, and ε on the replica dump (vacuum_replicas.py must have run) — the
absolute error of the arm's score against the exact one, noise removed. The
dump keeps the coefficients and the per-config (e, S) so train_vacuum.py and
the readout can pair every network against every arm on the same replicas.

Env / --vac-*: VAC_ENSEMBLE_SEED, VAC_FIT_CONFIGS (300), VAC_VAL_CONFIGS (60),
VAC_BB_R (3), VAC_RIDGE (1e-10, relative to the mean diagonal), VAC_EVAL_ONLY,
VAC_SMOKE.

VAC_EVAL_ONLY=1 rescores an existing dump on the current replicas: A, b and the
loss sums are read back instead of accumulated (the accumulation is the 7 h of
run 1, not the ~30 min once quoted), the coefficients are re-solved from them —
the same numbers — and only ε and the per-config (e, S) are new. It refuses a
dump written with other fit/val configurations or another BB_R.

    python scripts/vacuum_classical.py
    VAC_EVAL_ONLY=1 python scripts/vacuum_classical.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vacuum_common as vc  # noqa: E402

vc.validate_argv("eval-only")

import torch  # noqa: E402

from gelt.vacuum import basis_gradients, spatial_slices, temporal_force  # noqa: E402

N_FIT = int(vc.cfg("VAC_FIT_CONFIGS", 10 if vc.SMOKE else 300))
N_VAL = int(vc.cfg("VAC_VAL_CONFIGS", 4 if vc.SMOKE else 60))
RIDGE = float(vc.cfg("VAC_RIDGE", 1e-10))
EVAL_ONLY = vc.flag("VAC_EVAL_ONLY")


def accumulate(configs, idx, label):
    K = len(vc.UNION_LABELS)
    A = torch.zeros(K, K, dtype=torch.float64)
    b = torch.zeros(K, dtype=torch.float64)
    FF, ncomp = 0.0, 0
    t = time.time()
    for k, i in enumerate(idx):
        U4 = configs[i][None].to(vc.device, torch.complex128)
        F = temporal_force(U4, vc.G, vc.BETA, vc.XI).flatten(1)  # (Lt, n)
        U3 = spatial_slices(U4)
        g = basis_gradients(U3, vc.union_ops).flatten(2)  # (K, Lt, n)
        gf = g.reshape(K, -1)
        A += (gf @ gf.T).cpu()
        b += (gf @ F.reshape(-1)).cpu()
        FF += float((F * F).sum())
        ncomp += F.numel()
        if (k + 1) % 25 == 0:
            print(f"  {label}: {k + 1}/{len(idx)}  {(time.time() - t) / (k + 1):.2f} s/config",
                  flush=True)
    return A, b, FF, ncomp


def solve(A, b, idx):
    As = A[idx][:, idx]
    lam = RIDGE * float(As.diagonal().mean())
    return torch.linalg.solve(As + lam * torch.eye(len(idx), dtype=As.dtype), b[idx])


def loss(A, b, FF, n, idx, c):
    As, bs = A[idx][:, idx], b[idx]
    return float(c @ As @ c - 2 * c @ bs + FF) / n


def main():
    vc.require_cuda("vacuum_classical")
    configs = vc.load_ensemble()
    train, val, _ = vc.splits(configs.shape[0])
    fit_idx, val_idx = vc.evenly(train, N_FIT), vc.evenly(val, N_VAL)
    print(f"union basis: {len(vc.UNION_LABELS)} operators | fit on {len(fit_idx)} train "
          f"configs × {vc.LT} slices | val {len(val_idx)} configs")
    if EVAL_ONLY:
        if not os.path.exists(vc.classical_path()):
            raise SystemExit(f"VAC_EVAL_ONLY needs {vc.classical_path()}")
        old = torch.load(vc.classical_path(), weights_only=False)
        om = old["meta"]
        if (list(om["fit"]) != list(fit_idx) or list(om["val"]) != list(val_idx)
                or om["bb_r"] != vc.BB_R or old["labels"] != vc.UNION_LABELS):
            raise SystemExit(f"{vc.classical_path()} was accumulated on other configurations "
                             "or another basis — rerun without VAC_EVAL_ONLY")
        A, b, FF, n = old["A"], old["b"], old["FF"], old["n"]
        Av, bv, FFv, nv = old["Av"], old["bv"], old["FFv"], old["nv"]
        print(f"eval-only: A, b read back from {vc.classical_path()}")
    else:
        A, b, FF, n = accumulate(configs, fit_idx, "fit")
        Av, bv, FFv, nv = accumulate(configs, val_idx, "val")

    rep = None
    if os.path.exists(vc.replica_path()):
        rep = torch.load(vc.replica_path(), weights_only=False)
        U3s = torch.stack(rep["U3"]).to(vc.device, torch.complex128)  # (n, 3, *Λ, 2, 2)
        grads = []
        for j in range(U3s.shape[0]):
            grads.append(basis_gradients(U3s[j:j + 1], vc.union_ops)[:, 0].cpu())
        G_rep = torch.stack(grads)  # (n_rep, K, 3, *Λ, 3)
    else:
        print(f"** no replica dump at {vc.replica_path()} — ε is skipped (run "
              "vacuum_replicas.py first)")

    out = dict(labels=vc.UNION_LABELS, A=A, b=b, FF=FF, n=n, Av=Av, bv=bv, FFv=FFv, nv=nv,
               arms={}, meta=dict(fit=fit_idx, val=val_idx, bb_r=vc.BB_R, ridge=RIDGE,
                                  ensemble_seed=vc.ENSEMBLE_SEED, smoke=vc.SMOKE))
    null = dict(train=FF / n, val=FFv / nv)
    print(f"\nnull Φ ≡ 0: train {null['train']:.4f}  val {null['val']:.4f} per component")
    if rep is not None:
        S = [vc.per_config_error(torch.zeros_like(f, dtype=torch.float64), rep, j)[1]
             for j, f in enumerate(rep["F0"])]
        out["S"] = S
        out["rep_idx"] = list(rep["idx"])
        out["rep_tag"] = vc.replica_tag(rep)
    print(f"{'arm':10s} {'K':>3s}  {'train':>8s}  {'val':>8s}   ε (exact score, noise removed)")
    for arm in vc.ARM_ORDER:
        idx = vc.arm_index(arm)
        c = solve(A, b, idx)
        r = dict(coef=c, labels=vc.ARMS[arm], train=loss(A, b, FF, n, idx, c),
                 val=loss(Av, bv, FFv, nv, idx, c))
        line = f"{arm:10s} {len(idx):3d}  {r['train']:8.4f}  {r['val']:8.4f}"
        if rep is not None:
            e = []
            for j in range(G_rep.shape[0]):
                g = torch.einsum("k,k...->...", c, G_rep[j][idx])
                e.append(vc.per_config_error(g, rep, j)[0])
            r["e"] = e
            r["eps"] = vc.jackknife_ratio(e, S, vc.JACK_BLOCK)
            line += f"   {r['eps'][0]:.4f} ± {r['eps'][1]:.4f}"
        out["arms"][arm] = r
        print(line)
    if rep is not None:
        print("\npaired Δε (same replicas):")
        for a, bname in (("dr", "shapes"), ("shapes", "shapes_sm"), ("dr", "bb"),
                         ("shapes_sm", "full"), ("bb", "full")):
            d = vc.jackknife_diff(out["arms"][a]["e"], out["arms"][bname]["e"], S, vc.JACK_BLOCK)
            print(f"  ε({a}) − ε({bname}) = {d[0]:+.4f} ± {d[1]:.4f}")
    print("\n`full` coefficients:")
    for l, v in zip(vc.ARMS["full"], out["arms"]["full"]["coef"].tolist()):
        print(f"  {l:10s} {v:+.5g}")
    torch.save(out, vc.classical_path())
    print(f"→ {vc.classical_path()}")


if __name__ == "__main__":
    main()
