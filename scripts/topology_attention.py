"""Does the 4D q(x) network's attention localize on topological structure?

Clause 1 of the conference abstract (notes/topological_localization.md). The
network trained by ``train_gelt.py`` is read out the way the glueball operator
was: attention weights are gauge invariants, hence a measurable field.

The script answers the **pivot question first**, because it may make everything
after it moot. Naive q(x) is quadratic in the plaquettes *at x*, so zero
receptive field is needed to solve the task exactly (R² = 1 confirms the
network found that solution). If the attention collapsed to pure
self-attention there is no range to correlate with anything, and the honest
move is to retrain on *cooled* q(x) — a genuinely non-local target — rather
than to keep analysing a degenerate map. The collapse diagnostic is printed
before the expensive ablation arm, which is skipped when it fires.

Ground truth is the **cooled** charge density (n_cool from the
check_cooling.py pre-flight), a field the network was never trained on — so a
correlation with it is an emergent finding, not a restatement of the target.
The naive charge's Z < 1 normalization is irrelevant: Spearman is rank-based.

Run (on the box holding the checkpoint):
    python scripts/topology_attention.py [checkpoint]

Writes ``topology_attention.png`` and ``datasets/topology_attention_stats.pt``.
"""

import functools
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visualize_glueball_attention import spearman_per_slice  # noqa: E402

from gelt.blocks_rope import GELT  # noqa: E402
from gelt.lattice import (  # noqa: E402
    SU,
    build_transport_average,
    plaquette_tensor,
    topological_charge_density,
)
from gelt.sampler import heatbath_overrelaxation_sweep, mcmc_ensemble  # noqa: E402
from gelt.topology import cool  # noqa: E402


# ── Tunables (must match train_gelt.py's physics block) ───────────────────────
L, D, R = 8, 4, 2
BETA = 2.4
NHEAD, LAYERS, D_QKV, D_MODEL, MLP_HIDDEN = 2, 4, 8, 16, 32
gaugegroup = SU(2)
MODEL_DTYPE = torch.complex64

N_VIZ = 32  # fresh held-out configs; its own seed ⇒ independent chain
VIZ_SEED = 11  # ≠ the training seed (0)
N_COOL = 35  # from scripts/check_cooling.py: Q plateaus, |Q| still stable
N_ABL = N_VIZ  # ablation sample = the jackknife's sample size
# Below this mean ℓ_att the attention carries no range worth analysing and the
# ablation arm is skipped. One offset shell is |Δx|=1, so 0.05 is "essentially
# all the mass sits on the site itself".
COLLAPSE_THRESHOLD = 0.05

CKPT = sys.argv[1] if len(sys.argv) > 1 else f"best_gelt_topo_L{L}_b{BETA}_R{R}.pth"
CACHE = f"datasets/topo_viz_configs_L{L}_b{BETA}_N{N_VIZ}_seed{VIZ_SEED}.pt"
STATS = "datasets/topology_attention_stats.pt"

# cuda → cpu: cooling projects onto the group with a complex SVD, which MPS
# cannot do.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensemble():
    if os.path.exists(CACHE):
        print(f"Loading cached {CACHE}")
        return torch.load(CACHE)
    print(f"Sampling N={N_VIZ} held-out configs (seed {VIZ_SEED}) …")
    torch.manual_seed(VIZ_SEED)
    configs, acc = mcmc_ensemble(
        L=L, D=D, gaugegroup=gaugegroup, beta=BETA, n_configs=N_VIZ,
        n_therm=200, n_skip=5, progress=True,
        sweep_fn=functools.partial(heatbath_overrelaxation_sweep, n_or=4),
    )
    print(f"  acceptance = {acc:.2f}")
    os.makedirs("datasets", exist_ok=True)
    torch.save(configs, CACHE)
    return configs


def build_model():
    model = GELT(
        gaugegroup=gaugegroup, L=L, D=D, R=R, nhead=NHEAD, gemhsa_layers=LAYERS,
        d_qkv=D_QKV, gate="softplus", dtype=MODEL_DTYPE, mlp_hidden=MLP_HIDDEN,
        mlp_out=1, reduction="none", init_scale=10, qk_init_scale=1.0,
        mlp_zero_init=True, d_model=D_MODEL, grad_checkpoint=False,
    ).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
    model.eval()
    print(f"loaded {CKPT}")
    return model


@torch.no_grad()
def forward_one(model, U1, collect=True):
    """One config → per-site readout and (optionally) the per-layer α stack."""
    W = plaquette_tensor(U1, gaugegroup)
    T = build_transport_average(U1, R, gaugegroup)
    out = model(W.to(device), T.to(device))
    A = (
        [layer._last_alpha.cpu() for layer in model.gemhsa_models]
        if collect
        else None
    )
    return out.cpu(), A


def main():
    print(f"device: {device}")
    configs = ensemble().to(MODEL_DTYPE)
    model = build_model()

    offsets = model.gemhsa_models[0].offsets  # index 0 = self (Δx = 0)
    n_off = len(offsets)
    dist = torch.tensor([sum(abs(c) for c in o) for o in offsets], dtype=torch.float32)
    axis_idx = [
        [offsets.index(tuple(s * int(d == k) for k in range(D))) for s in (+1, -1)]
        for d in range(D)
    ]

    # ── Ground truth: cooled |q(x)|, a field the network never saw.
    print(f"Cooling {len(configs)} configs × {N_COOL} steps …")
    q_cool = topological_charge_density(
        cool(configs.to(device), gaugegroup, n_steps=N_COOL, progress=True), gaugegroup
    ).cpu()
    Q = q_cool.flatten(start_dim=1).sum(dim=1)
    n_charged = int((Q.abs() >= 0.5).sum())
    print(f"  ⟨|Q|⟩ = {Q.abs().mean():.3f} | configs with |Q| ≥ 0.5: {n_charged}/{len(Q)}")
    if n_charged < len(Q) // 4:
        print("  WARNING: most configs are in the Q≈0 sector — few lumps to localize on.")

    # ── Extraction.
    ell_all, ent_all, preds = [], [], []
    sum_a = torch.zeros(LAYERS, NHEAD, n_off, dtype=torch.float64)
    axis_counts = torch.zeros(LAYERS, NHEAD, D, dtype=torch.float64)
    n_sites_tot = 0
    for c in tqdm(range(len(configs)), desc="extract"):
        out, A = forward_one(model, configs[c : c + 1])
        preds.append(out.reshape(-1))
        ells, ents = [], []
        for l, alpha in enumerate(A):  # (1, H, n_off, *Λ)
            a = alpha[0]
            sum_a[l] += a.double().flatten(2).sum(dim=2)  # (H, n_off)
            ells.append((a * dist.view(1, n_off, *([1] * D))).sum(dim=1).flatten(1))
            ents.append(-(a.clamp_min(1e-12).log() * a).sum(dim=1).flatten(1))
            am = torch.stack([a[:, idx].sum(dim=1) for idx in axis_idx])  # (D, H, *Λ)
            pick = am.argmax(dim=0).flatten(1)
            for d in range(D):
                axis_counts[l, :, d] += (pick == d).sum(dim=1).double()
        ell_all.append(torch.stack(ells))  # (LAYERS, H, n_sites)
        ent_all.append(torch.stack(ents))
        n_sites_tot += ell_all[-1].shape[-1]

    ell = torch.stack(ell_all, dim=2)  # (LAYERS, H, n_cfg, n_sites)
    ent = torch.stack(ent_all, dim=2)
    mean_a = (sum_a / n_sites_tot).float()
    axis_frac = axis_counts / axis_counts.sum(dim=2, keepdim=True)

    # ── THE PIVOT QUESTION, printed before anything expensive.
    ell_mean = ell.mean(dim=(2, 3))
    ell_std = ell.std(dim=(2, 3))
    self_w = mean_a[:, :, 0]
    print("\nlayer head   ℓ_att            self-α   entropy   axis split")
    for l in range(LAYERS):
        for h in range(NHEAD):
            print(
                f"  {l + 1}     {h}   {ell_mean[l, h]:.4f}±{ell_std[l, h]:.4f}"
                f"   {self_w[l, h]:.4f}   {ent[l, h].mean():.3f}"
                f"   {'/'.join(f'{f:.2f}' for f in axis_frac[l, h])}"
            )
    collapsed = bool(ell_mean.max() < COLLAPSE_THRESHOLD)
    print(
        f"\nmax ℓ_att over all heads = {ell_mean.max():.4f} "
        f"(threshold {COLLAPSE_THRESHOLD})"
    )
    if collapsed:
        print(
            "COLLAPSED: the attention is pure self-attention. This is the\n"
            "expected outcome for naive q(x) (on-site quadratic), NOT a bug.\n"
            "PIVOT: retrain on cooled q(x), which cannot be solved on-site.\n"
            "Skipping the ablation arm — there is nothing to ablate."
        )
    else:
        print("NOT collapsed: the attention carries range. Proceeding.")

    # ── Ground-truth correlation, per (layer, head): ℓ_att(x) vs |q_cool(x)|.
    qflat = q_cool.abs().flatten(1)  # (n_cfg, n_sites)
    spear = torch.zeros(LAYERS, NHEAD)
    spear_e = torch.zeros(LAYERS, NHEAD)
    for l in range(LAYERS):
        for h in range(NHEAD):
            r, e, _ = spearman_per_slice(ell[l, h], qflat, len(configs))
            spear[l, h], spear_e[l, h] = r, e
            print(f"  Spearman(ℓ_att, |q_cool|) L{l + 1}h{h} = {r:+.3f} ± {e:.3f}")

    # ── Intervention (skipped on collapse): correlated delete-one jackknife of
    # the MSE degradation when a head's w_mix column is zeroed.
    # The ablation must score the model on the task it was TRAINED on — naive
    # q(x), not the cooled field (which is only the localization ground truth
    # and which the network has never seen). train_gelt.py standardizes its
    # target, so the target is standardized here too; μ_y ≈ 0 and σ_y is a
    # property of the ensemble, so the viz set's own scaler matches the
    # training one to well within the precision this comparison needs. Only the
    # target is standardized — rescaling an ablated prediction would hide
    # exactly the damage being measured.
    q_naive = topological_charge_density(configs.to(device), gaugegroup).cpu()
    y = q_naive.reshape(-1)
    y = (y - y.mean()) / y.std().clamp_min(1e-12)
    base = torch.cat(preds).real
    abl = torch.zeros(LAYERS, NHEAD, dtype=torch.float64)
    abl_err = torch.zeros_like(abl)

    def per_config_mse(p):
        return ((p - y) ** 2).reshape(len(configs), -1).mean(dim=1).double()

    if not collapsed:
        base_cfg = per_config_mse(base)
        for l in tqdm(range(LAYERS), desc="ablate"):
            layer = model.gemhsa_models[l]
            for h in range(NHEAD):
                orig = layer.w_mix[:, h].detach().clone()
                with torch.no_grad():
                    layer.w_mix[:, h] = 0
                p = torch.cat(
                    [forward_one(model, configs[c : c + 1], collect=False)[0].reshape(-1)
                     for c in range(N_ABL)]
                ).real
                with torch.no_grad():
                    layer.w_mix[:, h] = orig
                d = per_config_mse(p) - base_cfg
                n = len(d)
                jk = torch.tensor(
                    [d[torch.arange(n) != i].mean() for i in range(n)], dtype=torch.float64
                )
                abl[l, h] = d.mean()
                abl_err[l, h] = ((n - 1) / n * (jk - jk.mean()).pow(2).sum()).sqrt()
                print(f"  ablate L{l + 1}h{h}: ΔMSE = {abl[l, h]:+.3e} ± {abl_err[l, h]:.1e}")

    # ── Figure.
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for h in range(NHEAD):
        ax[0].errorbar(np.arange(1, LAYERS + 1) + 0.05 * h, ell_mean[:, h],
                       yerr=ell_std[:, h], fmt="o-", capsize=3, label=f"head {h}")
    ax[0].axhline(COLLAPSE_THRESHOLD, ls=":", color="red", label="collapse threshold")
    ax[0].set_xlabel("layer"); ax[0].set_ylabel(r"$\ell_{\rm att}$")
    ax[0].set_title("Attention range vs depth\n(the pivot question)")
    ax[0].legend(); ax[0].grid(True, alpha=0.3)

    dmask = [dist == d for d in range(int(dist.max()) + 1)]
    for l in range(LAYERS):
        for h in range(NHEAD):
            ax[1].plot(range(len(dmask)), [mean_a[l, h, m].sum() for m in dmask],
                       marker="o", label=f"L{l + 1}h{h}")
    ax[1].set_yscale("log"); ax[1].set_xlabel(r"$|\Delta x|_1$")
    ax[1].set_ylabel("mean attention mass")
    ax[1].set_title("Radial profile")
    ax[1].legend(fontsize=7, ncol=2); ax[1].grid(True, alpha=0.3)

    im = ax[2].imshow(spear.numpy(), cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    for l in range(LAYERS):
        for h in range(NHEAD):
            ax[2].text(h, l, f"{spear[l, h]:+.3f}\n±{spear_e[l, h]:.3f}",
                       ha="center", va="center", fontsize=9)
    ax[2].set_xticks(range(NHEAD), [f"head {h}" for h in range(NHEAD)])
    ax[2].set_yticks(range(LAYERS), [f"layer {l + 1}" for l in range(LAYERS)])
    ax[2].set_title(r"Spearman$(\ell_{\rm att}(x), |q_{\rm cool}(x)|)$")
    fig.colorbar(im, ax=ax[2], shrink=0.8)

    fig.suptitle(
        f"Topological localization of attention — SU(2) {L}⁴ β={BETA}, "
        f"N={len(configs)} held out, n_cool={N_COOL}"
        + ("   [ATTENTION COLLAPSED]" if collapsed else ""),
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig("topology_attention.png", dpi=130, bbox_inches="tight")
    print("Saved topology_attention.png")

    torch.save(
        {
            "offsets": offsets, "mean_alpha": mean_a, "ell_mean": ell_mean,
            "ell_std": ell_std, "axis_frac": axis_frac, "spearman": spear,
            "spearman_err": spear_e, "ablation": abl, "ablation_err": abl_err,
            "collapsed": collapsed, "Q": Q, "n_charged": n_charged,
            "meta": {"L": L, "beta": BETA, "R": R, "n_cool": N_COOL,
                     "checkpoint": CKPT, "viz_seed": VIZ_SEED},
        },
        STATS,
    )
    print(f"Saved {STATS}")


if __name__ == "__main__":
    main()
