"""The vacuum wavefunctional by force matching: a network, residual over a classical arm.

    log ψ₀²(U) = −S_s(U) + Φ_arm(U) + Φ_net(U),

Φ_arm the force-matching least-squares fit of a classical arm (vacuum_classical.py,
frozen) and Φ_net = Σ_x net(U)(x) the per-site output of the glueball study's
network summed over the slice. The loss is the force-matching error

    mean over links and colour components of ( ∇Φ_net − [F_temp − ∇Φ_arm] )²,

with ∇ the Lie gradient taken through the plaquettes, the smearing and the
transport (double backprop). Its minimiser over all functions is the exact
vacuum (gelt/vacuum.py). The readout is zero-initialised, so a run starts at the
classical arm's ε and any gain is, by construction, structure that arm lacks.

Selection: lowest validation force-matching loss (val split, fixed slices).
Readout: ε on the replica dump, paired against the residual arm on the same
configurations (vacuum_readout.py reads the dumps).

Env / --vac-*: VAC_ARCH (gelt | lcnn), VAC_ALPHA_MODE (softmax | frozen),
VAC_INPUT_LEVELS ("0"; e.g. "0,2" — each level is APE inside the gradient, so
it costs), VAC_RESIDUAL (full | shapes_sm | dr | … | none), VAC_INIT_SEED (0),
VAC_LR (1e-3), VAC_WARMUP (100 steps), VAC_EPOCHS (12), VAC_PATIENCE (4),
VAC_BATCH (4 slices per step, one config), VAC_TRAIN_CONFIGS (all),
VAC_VAL_CONFIGS (40), VAC_TAG, VAC_EVAL_ONLY, VAC_SMOKE.

    python scripts/train_vacuum.py
    VAC_ALPHA_MODE=frozen python scripts/train_vacuum.py
"""

import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vacuum_common as vc  # noqa: E402

vc.validate_argv("eval-only")

import torch  # noqa: E402

from gelt.vacuum import spatial_slices, temporal_force  # noqa: E402

C = vc.net_config()
LR = float(vc.cfg("VAC_LR", 1e-3))
WARMUP = int(vc.cfg("VAC_WARMUP", 5 if vc.SMOKE else 100))
EPOCHS = int(vc.cfg("VAC_EPOCHS", 2 if vc.SMOKE else 12))
PATIENCE = int(vc.cfg("VAC_PATIENCE", 4))
BATCH = int(vc.cfg("VAC_BATCH", 2 if vc.SMOKE else 4))
N_TRAIN = int(vc.cfg("VAC_TRAIN_CONFIGS", 8 if vc.SMOKE else 10**9))
N_VAL = int(vc.cfg("VAC_VAL_CONFIGS", 3 if vc.SMOKE else 40))
EVAL_ONLY = vc.flag("VAC_EVAL_ONLY")
STEM = vc.net_stem(C)
VAL_SLICES = [0, vc.LT // 4, vc.LT // 2, 3 * vc.LT // 4]


def residual():
    if C["residual"] == "none":
        return None, None
    if not os.path.exists(vc.classical_path()):
        raise SystemExit(f"VAC_RESIDUAL={C['residual']} needs {vc.classical_path()} "
                         "(run vacuum_classical.py), or VAC_RESIDUAL=none")
    cl = torch.load(vc.classical_path(), weights_only=False)
    if C["residual"] not in cl["arms"]:
        raise SystemExit(f"no arm {C['residual']!r} in {vc.classical_path()}")
    return cl, cl["arms"][C["residual"]]["coef"]


def targets(U4, coef, sel):
    """Slices ``sel`` of one config and their residual targets F − ∇Φ_arm."""
    U4c = U4[None].to(vc.device, torch.complex128)
    F = temporal_force(U4c, vc.G, vc.BETA, vc.XI)[sel]
    U3 = spatial_slices(U4c)[sel]
    if coef is not None:
        F = F - vc.classical_grad(C["residual"], coef, U3)
    return U3.to(vc.tg.MODEL_DTYPE), F.float()


def val_loss(model, val_set):
    tot, n = 0.0, 0
    for U3, T in val_set:
        g = vc.net_grad(model, C, U3.to(vc.device))
        tot += float(((g - T.to(vc.device)) ** 2).sum())
        n += T.numel()
    return tot / n


def train(model, configs, train_idx, val_set, coef):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / WARMUP))
    gen = torch.Generator().manual_seed(C["init_seed"])
    best, bad, hist = val_loss(model, val_set), 0, []
    print(f"start (Φ_net ≡ 0): val {best:.5f}")
    torch.save(dict(state=model.state_dict(), config=C, val=best, epoch=0), STEM + ".pth")
    for ep in range(EPOCHS):
        model.train()
        t, run, steps = time.time(), 0.0, 0
        order = [train_idx[int(k)] for k in torch.randperm(len(train_idx), generator=gen)]
        for i in order:
            sel = torch.randperm(vc.LT, generator=gen)[:BATCH].tolist()
            U3, T = targets(configs[i], coef, sel)
            g = vc.net_grad(model, C, U3, create_graph=True)
            loss = ((g - T) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += loss.item()
            steps += 1
            if steps % 100 == 0:
                print(f"  ep {ep + 1} step {steps}/{len(order)}  train {run / steps:.5f}  "
                      f"{(time.time() - t) / steps:.2f} s/step", flush=True)
        model.eval()
        v = val_loss(model, val_set)
        hist.append(dict(epoch=ep + 1, train=run / max(steps, 1), val=v,
                         seconds=time.time() - t))
        flag = ""
        if v < best:
            best, bad, flag = v, 0, "  *"
            torch.save(dict(state=model.state_dict(), config=C, val=v, epoch=ep + 1),
                       STEM + ".pth")
        else:
            bad += 1
        print(f"epoch {ep + 1}: train {run / max(steps, 1):.5f}  val {v:.5f}{flag}  "
              f"({time.time() - t:.0f} s)", flush=True)
        if not math.isfinite(v) or bad >= PATIENCE:
            break
    return hist


def evaluate(model, coef, cl):
    if not os.path.exists(vc.replica_path()):
        print(f"** no replica dump at {vc.replica_path()} — ε skipped")
        return None
    rep = torch.load(vc.replica_path(), weights_only=False)
    e, S = [], []
    for j, U3 in enumerate(rep["U3"]):
        U3 = U3[None].to(vc.device)
        g = vc.net_grad(model, C, U3)[0].double()
        if coef is not None:
            g = g + vc.classical_grad(C["residual"], coef, U3.to(torch.complex128))[0]
        ej, sj = vc.per_config_error(g.cpu(), rep, j)
        e.append(ej)
        S.append(sj)
    out = dict(e=e, S=S, rep_idx=list(rep["idx"]), rep_tag=vc.replica_tag(rep),
               eps=vc.jackknife_ratio(e, S, vc.JACK_BLOCK))
    print(f"ε(net + {C['residual']}) = {out['eps'][0]:.4f} ± {out['eps'][1]:.4f}")
    if cl is not None:
        if not vc.scored_on(cl, rep):
            print("** the classical dump was evaluated on other replicas — no pairing "
                  "(VAC_EVAL_ONLY=1 python scripts/vacuum_classical.py rescores it)")
        else:
            arm = cl["arms"][C["residual"]]
            d = vc.jackknife_diff(arm["e"], e, S, vc.JACK_BLOCK)
            out["gain_vs_residual"] = d
            print(f"ε({C['residual']}) − ε(net + {C['residual']}) = {d[0]:+.4f} ± {d[1]:.4f}"
                  f"   ({d[0] / max(d[1], 1e-12):+.1f}σ)")
    return out


def main():
    vc.require_cuda("train_vacuum")
    print(f"config {C} | lr {LR} | batch {BATCH} | stem {STEM}")
    configs = vc.load_ensemble()
    train_idx, val_r, _ = vc.splits(configs.shape[0])
    train_idx = vc.evenly(train_idx, N_TRAIN)
    cl, coef = residual()
    model = vc.build_net(C)
    n_par = sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())
    print(f"{C['arch']}: {n_par} real parameters")
    hist = []
    if EVAL_ONLY and os.path.exists(STEM + ".pt"):
        # a rescore must not erase the training record it is rescoring
        hist = torch.load(STEM + ".pt", weights_only=False).get("history", [])
    if not EVAL_ONLY:
        val_set = [targets(configs[i], coef, VAL_SLICES) for i in vc.evenly(val_r, N_VAL)]
        val_set = [(U.cpu(), T.cpu()) for U, T in val_set]
        hist = train(model, configs, train_idx, val_set, coef)
    model, _ = vc.load_net(STEM + ".pth")
    ev = evaluate(model, coef, cl)
    torch.save(dict(config=C, history=hist, eval=ev, lr=LR, batch=BATCH,
                    checkpoint=STEM + ".pth", ensemble_seed=vc.ENSEMBLE_SEED,
                    smoke=vc.SMOKE), STEM + ".pt")
    print(f"→ {STEM}.pt")


if __name__ == "__main__":
    main()
