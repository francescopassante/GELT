"""Phase B: does the learned attention range track the correlation length?

The clause the conference abstract promises, run where it can actually be
asked. Phase A (scripts/z2_beta_scan.py) established that 3D Z₂ near its
second-order point gives ξ = 2.05 → 5.28 lattice spacings across
β ∈ [0.745, 0.760] — real dynamic range, unlike the SU(2) glueball where
ξ_s ≤ 1.1 everywhere and the question is unaskable.

This trains one variational operator per β, exactly as train_glueball.py does
for SU(2): a **per-timeslice** operator (here 2D, since 3D Z₂ has two spatial
directions) on the Rayleigh loss −mean_Δ C(Δ)/C(0), whose converged value is
the mass. Then it reads ℓ_att off the attention and writes one row per β.

Both axes of the final plot come from the same run: ξ from the classical scan
(and cross-checked against the loss), ℓ_att from the network. The output is
consumed by the plotting/analysis step, one row per β.

**R = 6 is the load-bearing choice.** ℓ_att is bounded by R, so R must exceed
ξ_max = 5.3 or the range saturates at its ceiling and we reproduce the failure
of the SU(2) study, where six of eight heads sat at ℓ_att = 2.000 ± 0.000 with
R = 2 and the statistic was censored rather than measured. In 2D the L1 ball at
R=6 is only 84 offsets, so the headroom is nearly free.

Run (one β per invocation, so phases are failure-isolated):
    python scripts/train_z2_glueball.py 0.756

Writes ``best_z2_glueball_b<beta>.pth`` and appends a row to
``datasets/z2_attention_rows.pt``.
"""

import os
import sys

import numpy as np
import torch
from tqdm import tqdm

from gelt.blocks_rope import GELT
from gelt.glueball import ape_smear
from gelt.lattice import Z2, build_transport_average, plaquette_tensor
from gelt.sampler import mcmc_ensemble, z2_heatbath_sweep


# ── Tunables (lattice block MUST match z2_beta_scan.py: same cache key) ───────
L, D, LT = 24, 3, 48
N_CONFIGS, N_THERM, N_SKIP = 2000, 500, 200
# Configurations actually used for training. The precomputed transport is
# ~9.3 MB per configuration (48 slices × 84 offsets × 24² floats), so all 2000
# would need ~19 GB of host RAM. 800 configs is 38,400 timeslice samples for a
# ~10k-parameter model — statistics are nowhere near the binding constraint.
# Raise it if the box has the memory and the fit looks statistics-limited.
N_USE = 800
PREP_CHUNK = 8  # configs per precompute batch (peak GPU transient)
SMEAR_ALPHA = 0.5
gaugegroup = Z2()
NC = gaugegroup.nc

BETA = float(sys.argv[1]) if len(sys.argv) > 1 else 0.756

# R must exceed ξ_max = 5.28 (see the module docstring): a range statistic
# bounded below the scale it is meant to track measures the bound, not the
# physics.
R = 6
GEMHSA_LAYERS = 4
NHEAD = 2
D_QKV = 8  # ≥ 2D = 4 for full RoPE axis coverage, with margin
D_MODEL = 16
MLP_HIDDEN = 32
INIT_SCALE, QK_INIT_SCALE = 10.0, 1.0
GATE = "softplus"
MODEL_DTYPE = torch.float32  # Z₂ is real; nc = 1

# In 2D there is exactly ONE plaquette plane, so each smearing level
# contributes a single channel (against 3 for the SU(2) 3D slices).
INPUT_SMEAR_LEVELS = (0, 4, 8, 16)
IN_CHANNELS = 1 * len(INPUT_SMEAR_LEVELS)

LR = 3e-3
WEIGHT_DECAY = 1e-3
EPOCHS = 40
PATIENCE = 8
BATCH_CONFIGS = 4
LOSS_DELTAS = (1, 2)
SCALE_REG = 1e-2  # (log C(0))² pin on the Ō → λŌ flat direction — without it
#                   the smeared, site-coherent inputs pump the operator scale
#                   until the loss overflows (the Run-5 lesson)
EPS = 1e-8
TRAIN_FRACTION, VAL_FRACTION = 0.7, 0.1

CACHE = f"datasets/z2_configs_L{L}_Lt{LT}_b{BETA}_N{N_CONFIGS}.pt"
CHECKPOINT = f"best_z2_glueball_b{BETA}.pth"
ROWS = "datasets/z2_attention_rows.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensemble():
    """The SAME cache z2_beta_scan.py wrote — never resample if it is there."""
    if os.path.exists(CACHE):
        print(f"loading cached {CACHE}")
        return torch.load(CACHE)
    print(f"sampling N={N_CONFIGS} at β={BETA} (Phase A's cache was absent) …")
    configs, _ = mcmc_ensemble(
        L=L, D=D, gaugegroup=gaugegroup, beta=BETA, n_configs=N_CONFIGS,
        n_therm=N_THERM, n_skip=N_SKIP, progress=True, Lt=LT,
        sweep_fn=z2_heatbath_sweep,
    )
    os.makedirs("datasets", exist_ok=True)
    torch.save(configs, CACHE)
    return configs


def config_inputs(U_batch):
    """Smeared plaquette channels W and transport T for a config minibatch.

    ``U_batch`` : ``(b, 3, Lt, L, L, 1, 1)``. Mirrors train_glueball.py's
    network_obar with one dimension removed: the spatial slice is 2D, so
    ``U[:, 1:]`` has two directions and the plaquette tensor has one plane.
    Time never enters the network — the transfer-matrix bound that makes the
    converged loss a mass depends on it.

    This is the whole cost of a step and NONE of it depends on the weights,
    which is why :func:`prepare` runs it once instead of every epoch.
    """
    b = U_batch.shape[0]
    U = U_batch.to(device)
    Ws, U2_first, done = [], None, 0
    for lvl in sorted(INPUT_SMEAR_LEVELS):
        if lvl > done:
            U = ape_smear(U, gaugegroup, alpha=SMEAR_ALPHA, n_steps=lvl - done)
            done = lvl
        Usp = U[:, 1:].movedim(2, 1).contiguous()  # (b, Lt, 2, L, L, nc, nc)
        U2 = Usp.reshape(b * LT, 2, L, L, NC, NC)
        if U2_first is None:
            U2_first = U2
        Ws.append(plaquette_tensor(U2, gaugegroup))  # (b·Lt, 1, L, L, nc, nc)
    return torch.cat(Ws, dim=1), build_transport_average(U2_first, R, gaugegroup)


def prepare(configs):
    """Precompute (W, T) for every configuration, once, held on the CPU.

    Z₂ has nc = 1, so every matmul in the smearing and the transport is 1×1 —
    there is no arithmetic to speak of, only kernel-launch overhead from
    ape_smear's Python loop over configs and directions (≈128 tiny launches per
    batch to reach smearing level 16). Recomputing that every epoch left the
    GPU at 0% utilization while it waited on dispatches.

    W and T are functions of the configuration alone, so they are computed once
    here and indexed thereafter; the training step becomes the model forward,
    which is the only part that actually wants a GPU.
    """
    Ws, Ts = [], []
    for i in tqdm(range(0, len(configs), PREP_CHUNK), desc="precompute W,T"):
        W, T = config_inputs(configs[i : i + PREP_CHUNK])
        Ws.append(W.cpu())
        Ts.append(T.cpu())
    W_all = torch.cat(Ws).view(len(configs), LT, *Ws[0].shape[1:])
    T_all = torch.cat(Ts).view(len(configs), LT, *Ts[0].shape[1:])
    gb = (W_all.numel() + T_all.numel()) * 4 / 1e9
    print(f"precomputed W {tuple(W_all.shape)} T {tuple(T_all.shape)} — {gb:.1f} GB on CPU")
    return W_all, T_all


def obar_from(model, W, T):
    """Zero-momentum Ō(t) from precomputed inputs. ``W``/``T``: ``(b, Lt, …)``."""
    b = W.shape[0]
    O = model(
        W.reshape(b * LT, *W.shape[2:]).to(device),
        T.reshape(b * LT, *T.shape[2:]).to(device),
    )
    return O.sum(dim=(1, 2)).view(b, LT)


def rayleigh_loss(Obar):
    """−mean_Δ C(Δ)/C(0) with the scale pin. C(0) is the ONLY denominator."""
    d = Obar - Obar.mean()
    C0 = (d * d).mean()
    ratios = [(d.roll(-dt, dims=1) * d).mean() / C0.clamp_min(EPS) for dt in LOSS_DELTAS]
    loss = -sum(ratios) / len(ratios)
    return loss + SCALE_REG * torch.log(C0.clamp_min(EPS)) ** 2


def attention_stats(model, W, T, n_viz=16):
    """ℓ_att per (layer, head), and its site-to-site spread, on held-out configs."""
    offsets = model.gemhsa_models[0].offsets
    dist = torch.tensor([sum(abs(c) for c in o) for o in offsets], dtype=torch.float32)
    n_off = len(offsets)
    ells = [[] for _ in model.gemhsa_models]
    with torch.no_grad():
        for c in tqdm(range(min(n_viz, len(W))), desc="attention"):
            obar_from(model, W[c : c + 1], T[c : c + 1])
            for l, layer in enumerate(model.gemhsa_models):
                a = layer._last_alpha.cpu()  # (b·Lt, H, n_off, L, L)
                ells[l].append(
                    (a * dist.view(1, 1, n_off, 1, 1)).sum(dim=2).flatten(2)
                )
    ell = torch.stack([torch.cat(e) for e in ells])  # (layers, b·Lt, H, n_sites)
    return ell.mean(dim=(1, 3)), ell.std(dim=(1, 3)), offsets  # (layers, H)


def main():
    print(f"device: {device} | β = {BETA} | R = {R}")
    configs = ensemble()[:N_USE].to(MODEL_DTYPE)
    W_all, T_all = prepare(configs)
    del configs
    n = W_all.shape[0]
    n_tr, n_va = int(TRAIN_FRACTION * n), int(VAL_FRACTION * n)
    # Contiguous, chain-ordered split — never shuffled: neighbouring configs are
    # autocorrelated, so a random split leaks train↔held correlations.
    sl_tr, sl_va, sl_te = slice(0, n_tr), slice(n_tr, n_tr + n_va), slice(n_tr + n_va, n)
    W_tr, T_tr = W_all[sl_tr], T_all[sl_tr]
    W_va, T_va = W_all[sl_va], T_all[sl_va]
    W_te, T_te = W_all[sl_te], T_all[sl_te]
    print(f"split: train {len(W_tr)} | val {len(W_va)} | test {len(W_te)}")

    model = GELT(
        gaugegroup=gaugegroup, L=L, D=2, R=R, nhead=NHEAD,
        gemhsa_layers=GEMHSA_LAYERS, d_qkv=D_QKV, gate=GATE, dtype=MODEL_DTYPE,
        mlp_hidden=MLP_HIDDEN, mlp_out=1, reduction="none",
        init_scale=INIT_SCALE, qk_init_scale=QK_INIT_SCALE,
        mlp_zero_init=False,  # zero init ⇒ exactly zero Rayleigh gradient
        d_model=D_MODEL, grad_checkpoint=True, in_channels=IN_CHANNELS,
    ).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()):,} | "
          f"offsets: {len(model.gemhsa_models[0].offsets)}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best, bad = float("inf"), 0
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(W_tr))
        tot = 0.0
        for i in tqdm(range(0, len(W_tr), BATCH_CONFIGS), desc=f"ep {ep}", leave=False):
            idx = perm[i : i + BATCH_CONFIGS]
            loss = rayleigh_loss(obar_from(model, W_tr[idx], T_tr[idx]))
            if not torch.isfinite(loss):
                print("  non-finite loss, skipping batch")
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([
                rayleigh_loss(
                    obar_from(model, W_va[i : i + BATCH_CONFIGS], T_va[i : i + BATCH_CONFIGS])
                ).item()
                for i in range(0, len(W_va), BATCH_CONFIGS)
            ]))
        print(f"  ep {ep:3d}  train {tot / max(1, len(W_tr) // BATCH_CONFIGS):.4f}  val {vl:.4f}")
        if vl < best:
            best, bad = vl, 0
            torch.save(model.state_dict(), CHECKPOINT)
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"early stop at epoch {ep}")
                break

    model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True))
    model.eval()
    # The converged Rayleigh ratio is a mass estimate: C(1)/C(0) = e^{−m}.
    with torch.no_grad():
        ob = torch.cat([
            obar_from(model, W_te[i : i + BATCH_CONFIGS], T_te[i : i + BATCH_CONFIGS]).cpu()
            for i in range(0, len(W_te), BATCH_CONFIGS)
        ])
    d = ob.double() - ob.double().mean()
    r1 = ((d.roll(-1, dims=1) * d).mean() / (d * d).mean()).item()
    m_net = -np.log(max(r1, 1e-12))
    print(f"\nbest val loss {best:.4f} | test C(1)/C(0) = {r1:.4f} → m·a ≈ {m_net:.4f}")

    ell_mean, ell_std, offsets = attention_stats(model, W_te, T_te)
    print("layer head   ℓ_att")
    for l in range(GEMHSA_LAYERS):
        for h in range(NHEAD):
            print(f"  {l + 1}     {h}   {ell_mean[l, h]:.3f} ± {ell_std[l, h]:.3f}")
    print(f"\nmax ℓ_att = {ell_mean.max():.3f} against the ceiling R = {R}"
          "  (at the ceiling ⇒ censored, raise R)")

    rows = torch.load(ROWS) if os.path.exists(ROWS) else {}
    rows[BETA] = {
        "beta": BETA, "m_net": m_net, "xi_net": 1.0 / m_net if m_net > 0 else float("inf"),
        "val_loss": best, "ell_mean": ell_mean, "ell_std": ell_std,
        "R": R, "n_offsets": len(offsets),
    }
    torch.save(rows, ROWS)
    print(f"appended β={BETA} → {ROWS}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
