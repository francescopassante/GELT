"""Study 3 (head/layer specialization) on the trained glueball operator.

First executable slice of the interpretability program (notes/explainability.md,
thesis ch. 8): the attention map of the Run-5 variational 0⁺⁺ operator is read
out as a *measurement*. The chapter's validation rules are baked in:

  * **Statistics, not impressions** — per-(layer, head) attention range
    ℓ_att = Σ_n α_n·|Δx_n|₁ (Study-1's definition, applied per site), offset
    entropy, and axial anisotropy form the specialization taxonomy; ℓ_att vs
    layer is the RG-coarsening test (growth with depth would be *learned*,
    unlike a CNN's).
  * **Static/dynamic decomposition** — the config-averaged profile is exactly
    the kind of static kernel a convolution could supply, so the site-to-site
    spread of ℓ_att(x) (and the per-offset std of α) is reported alongside
    every mean: the content-dependent residual is the claim.
  * **Ground truth** — per-timeslice Spearman correlation of ℓ_att(x) with the
    smeared spatial-plaquette density o(x) (the physical field this operator
    measures; the glueball-task stand-in for ch. 8's cooled q(x)). The rank
    correlation is formed per slice (pooling sites across slices would mix
    per-slice offsets into the ranks), but the *error* is config-level: slices
    within a config are correlated over ~1/m, so configs are the independent
    units and r̄ ± std(per-config means)/√N_configs is the quoted number.
  * **Intervention or it is correlation** — a head-ablation matrix: zero head
    h's ``w_mix`` column in layer l (the residual stream is untouched) and
    measure the Rayleigh-ratio degradation on held-out configs, with a
    correlated delete-one jackknife over configs so the small entries can be
    told from noise.

The 2000-config production cache lives on the V100 box, but attention
statistics don't need it: a small fresh ensemble at the identical
(L, Lt, β, ξ) sampler settings is sampled locally (~1 s/sweep on an M2 CPU,
so ~8 min for N=32), cached under ``datasets/``, and — being a fresh chain —
is held-out by construction.

Model/task constants are imported from ``train_glueball.py`` so the two
scripts cannot drift. Forward passes chunk the Lt timeslices (the model is
per-slice, so Ō concatenates exactly) to keep the gathered-K/V transient
small enough for a laptop CPU.

Run:
    python scripts/visualize_glueball_attention.py [checkpoint]

Writes ``glueball_attention.png`` (specialization + validation panels),
``glueball_attention_kernels.png`` (per-(layer, head) mean-α offset maps),
and ``results/attention/glueball_attention_stats.pt`` (raw statistics for the thesis
figures).
"""

import functools
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# train_glueball.py is a sibling script, not a package module; make its
# constants importable regardless of how this script is launched.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_glueball as tg  # noqa: E402  (path fix must precede the import)

from gelt.blocks_rope import GELT  # noqa: E402
from gelt.glueball import ape_smear  # noqa: E402
from gelt.lattice import build_transport_average, plaquette_tensor  # noqa: E402
from gelt.sampler import heatbath_overrelaxation_sweep, mcmc_ensemble  # noqa: E402

# Output artifacts are grouped by study under results/; create the dirs the
# first time this runs in a fresh clone (they hold generated files only).
for _d in ("results/sampler", "results/glueball", "results/attention",
          "results/wilson_regression", "datasets"):
    os.makedirs(_d, exist_ok=True)



# ── Tunables ──────────────────────────────────────────────────────────────────
VIZ_N = int(os.environ.get("VIZ_N_CONFIGS", 16))  # fresh viz-ensemble size
VIZ_SEED = int(os.environ.get("VIZ_SEED", 7))  # ≠ any training/replication seed
# Configs entering the head-ablation Rayleigh estimate. This is the jackknife's
# sample size, so it defaults to the whole ensemble: the ablation arm is the
# expensive one (n_layers·H forwards per config) but an unresolved Δloss is
# worth nothing, and four of eight cells came out *negative* at N_ABL = 8.
N_ABL = min(int(os.environ.get("VIZ_N_ABLATE", VIZ_N)), VIZ_N)
CHUNK = 8  # timeslices per forward — the gathered-K/V memory knob (~1 GiB here)
CACHE = (
    f"datasets/glueball_viz_configs_L{tg.L}_Lt{tg.LT}_b{tg.BETA}"
    f"_xi{tg.XI}_N{VIZ_N}_seed{VIZ_SEED}.pt"
)
CKPT = sys.argv[1] if len(sys.argv) > 1 else tg.CHECKPOINT
STATS_DUMP = "results/attention/glueball_attention_stats.pt"


# ── Per-config input pipeline (mirrors tg.network_obar, kept per-config) ──────
@torch.no_grad()
def config_inputs(U4):
    """Smeared-channel W, transport T, and smeared density o for ONE config.

    ``U4`` : ``(4, Lt, L, L, L, nc, nc)``. Returns ``W (Lt, 3·n_levels, L³…)``,
    ``T (Lt, n_off−1, L³…)`` and ``o (Lt, L, L, L)`` — the per-site smeared
    spatial-plaquette density Σ_planes Re Tr P / nc at the *most* smeared
    level, the ground-truth field for the localization correlation.
    """
    U = U4.unsqueeze(0)
    Ws, U3_first, done = [], None, 0
    for lvl in sorted(tg.INPUT_SMEAR_LEVELS):
        if lvl > done:
            U = ape_smear(U, tg.gaugegroup, alpha=tg.SMEAR_ALPHA, n_steps=lvl - done)
            done = lvl
        Usp = U[:, 1:].movedim(2, 1).contiguous()  # (1, Lt, 3, L,L,L, nc,nc)
        U3 = Usp.reshape(tg.LT, 3, tg.L, tg.L, tg.L, tg.NC, tg.NC)
        if U3_first is None:
            U3_first = U3  # least-smeared level supplies the transport
        Ws.append(plaquette_tensor(U3, tg.gaugegroup))
    W = torch.cat(Ws, dim=1)
    T = build_transport_average(U3_first, tg.R, tg.gaugegroup)
    o = W[:, -3:].diagonal(dim1=-2, dim2=-1).sum(-1).real.sum(dim=1) / tg.NC
    return W, T, o


@torch.no_grad()
def forward_config(model, W, T, device, collect=True):
    """Chunked forward over one config's Lt slices.

    Returns the per-site readout ``O (Lt, L, L, L)`` and, when ``collect``,
    the per-layer attention stack ``A[l] (Lt, H, n_off, L, L, L)`` reassembled
    from each chunk's ``_last_alpha`` stash.
    """
    Os, alphas = [], [[] for _ in model.gemhsa_models]
    for i in range(0, W.shape[0], CHUNK):
        O = model(W[i : i + CHUNK].to(device), T[i : i + CHUNK].to(device))
        Os.append(O.cpu())
        if collect:
            for l, layer in enumerate(model.gemhsa_models):
                alphas[l].append(layer._last_alpha.cpu())
    A = [torch.cat(a) for a in alphas] if collect else None
    return torch.cat(Os), A


def rayleigh_ratios(Obar):
    """C(Δ)/C(0) for Δ ∈ LOSS_DELTAS on ``(B, Lt)`` — the pure variational
    part of tg.rayleigh_loss (the scale pin is training-only and would muddy
    the ablation comparison)."""
    d = Obar.double() - Obar.double().mean()
    C0 = (d * d).mean()
    return {dt: ((d.roll(-dt, dims=1) * d).mean() / C0).item() for dt in tg.LOSS_DELTAS}


def rayleigh_loss_value(Obar):
    """The pure variational part of tg.rayleigh_loss on ``(B, Lt)``."""
    r = rayleigh_ratios(Obar)
    return -sum(r.values()) / len(r)


def jackknife_dloss(obar_abl, obar_base):
    """Delete-one jackknife of the ablation's Δ(Rayleigh loss).

    Both arrays are ``(n_cfg, Lt)`` **on the same configurations**, so deleting
    config i from both and differencing keeps the comparison correlated — the
    ablated and intact operators share the ensemble noise that dominates a
    Rayleigh ratio at this sample size, and only the difference is being asked
    to resolve. Without this the Δloss column carries no error at all and its
    small entries (which came out *negative* — a head whose removal apparently
    improves the operator) cannot be told from noise.
    """
    n = obar_abl.shape[0]
    full = rayleigh_loss_value(obar_abl) - rayleigh_loss_value(obar_base)
    keep = ~torch.eye(n, dtype=torch.bool)
    vals = torch.tensor(
        [
            rayleigh_loss_value(obar_abl[keep[i]]) - rayleigh_loss_value(obar_base[keep[i]])
            for i in range(n)
        ],
        dtype=torch.float64,
    )
    err = ((n - 1) / n * (vals - vals.mean()).pow(2).sum()).sqrt().item()
    return full, err


def spearman_per_slice(a, b, n_cfg):
    """Mean and error of the per-slice Spearman rank correlation.

    ``a, b`` : ``(S, n_sites)`` with ``S = n_cfg · Lt`` in config-major order.
    Each timeslice is one measurement of the site-level correlation (pooling
    sites across slices would mix per-slice offsets into the ranks), but
    timeslices are *not* independent units: slices within one configuration
    are correlated over ~1/m ≈ 2.7 in Δ — that correlation is precisely what
    this whole program measures. The independent unit is the configuration, so
    the quoted error is the spread of the per-config means over ``n_cfg``.
    The naive slice-level error is returned alongside to document the size of
    the mistake, not to be quoted.
    """
    ra = a.argsort(dim=1).argsort(dim=1).double()
    rb = b.argsort(dim=1).argsort(dim=1).double()
    ra = ra - ra.mean(dim=1, keepdim=True)
    rb = rb - rb.mean(dim=1, keepdim=True)
    r = (ra * rb).sum(dim=1) / (ra.norm(dim=1) * rb.norm(dim=1))
    per_cfg = r.reshape(n_cfg, -1).mean(dim=1)
    err_cfg = (per_cfg.std() / np.sqrt(n_cfg)).item()
    err_slice = (r.std() / np.sqrt(len(r))).item()
    return r.mean().item(), err_cfg, err_slice


def main():
    # CPU on purpose: the model is complex64 and MPS complex support is too
    # patchy to trust for a measurement pass; the whole extraction is minutes.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # ── Fresh visualization ensemble (cached). Same sampler settings as the
    # production ensemble; its own seed ⇒ an independent chain, so every
    # statistic below is held-out by construction.
    if os.path.exists(CACHE):
        print(f"Loading cached viz ensemble {CACHE} …")
        configs = torch.load(CACHE)
    else:
        print(f"Sampling viz ensemble (N={VIZ_N}, seed={VIZ_SEED}) …")
        torch.manual_seed(VIZ_SEED)
        np.random.seed(VIZ_SEED)
        sweep = functools.partial(heatbath_overrelaxation_sweep, n_or=tg.N_OR, xi=tg.XI)
        configs, acc = mcmc_ensemble(
            L=tg.L,
            D=tg.D,
            gaugegroup=tg.gaugegroup,
            beta=tg.BETA,
            n_configs=VIZ_N,
            n_therm=tg.N_THERM,
            n_skip=tg.N_SKIP,
            sweep_fn=sweep,
            progress=True,
            Lt=tg.LT,
        )
        print(f"     acceptance = {acc:.2f}")
        os.makedirs("datasets", exist_ok=True)
        torch.save(configs, CACHE)
        print(f"     cached → {CACHE}")
    configs = configs.to(tg.MODEL_DTYPE)

    # ── Model: exactly train_glueball.py's construction, then the checkpoint.
    model = GELT(
        gaugegroup=tg.gaugegroup,
        L=tg.L,
        D=3,
        R=tg.R,
        nhead=tg.NHEAD,
        gemhsa_layers=tg.GEMHSA_LAYERS,
        d_qkv=tg.D_QKV,
        gate=tg.GATE,
        dtype=tg.MODEL_DTYPE,
        mlp_hidden=tg.MLP_HIDDEN,
        mlp_out=1,
        reduction="none",
        init_scale=tg.INIT_SCALE,
        qk_init_scale=tg.QK_INIT_SCALE,
        mlp_zero_init=False,
        d_model=tg.D_MODEL,
        grad_checkpoint=False,
        in_channels=3 * len(tg.INPUT_SMEAR_LEVELS),
    ).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
    model.eval()
    print(f"loaded {CKPT}")

    n_layers, H = tg.GEMHSA_LAYERS, tg.NHEAD
    offsets = model.gemhsa_models[0].offsets  # index 0 = self (Δx = 0)
    n_off = len(offsets)
    dist = torch.tensor([sum(abs(c) for c in o) for o in offsets], dtype=torch.float32)
    # Distance-1 offset indices per spatial axis (for the axial anisotropy).
    axis_idx = [
        [offsets.index(tuple(s * int(d == k) for k in range(3))) for s in (+1, -1)]
        for d in range(3)
    ]

    # ── Extraction pass: accumulate α statistics over every config × slice.
    sum_a = torch.zeros(n_layers, H, n_off, dtype=torch.float64)
    sumsq_a = torch.zeros_like(sum_a)
    n_samples = 0  # site-level samples per (layer, head, offset)
    ell_all = [[] for _ in range(n_layers)]  # per-slice per-site range ℓ(x)
    ent_sum = torch.zeros(n_layers, H, dtype=torch.float64)
    ent_sumsq = torch.zeros_like(ent_sum)
    aniso_sum = torch.zeros(n_layers, H, dtype=torch.float64)
    # Which spatial axis each head selects, per site. The per-site anisotropy
    # says a head concentrates its |Δx|=1 mass on ONE axis; it cannot say
    # whether that axis is the same everywhere. A flat histogram (1/3 each)
    # means the choice is content-driven and isotropy is restored in the
    # zero-momentum sum; a peaked one means a globally fixed axis, i.e. the
    # operator is not a cubic-group scalar and carries spin-2 admixture.
    axis_counts = torch.zeros(n_layers, H, 3, dtype=torch.float64)
    o_all = []
    obars = []
    field_ell = None  # (n_layers, H, L, L, L) ℓ(x) of config 0, t = 0 (map panel)
    field_o = None

    for c in tqdm(range(configs.shape[0]), desc="extract"):
        W, T, o = config_inputs(configs[c])
        O, A = forward_config(model, W, T, device)
        obars.append(O.sum(dim=(1, 2, 3)))
        o_all.append(o.reshape(tg.LT, -1))
        for l, alpha in enumerate(A):  # (Lt, H, n_off, L, L, L)
            a64 = alpha.double()
            sum_a[l] += a64.sum(dim=(0, 3, 4, 5))
            sumsq_a[l] += a64.pow(2).sum(dim=(0, 3, 4, 5))
            ell = (alpha * dist.view(1, 1, n_off, 1, 1, 1)).sum(dim=2)
            ell_all[l].append(ell.reshape(tg.LT, H, -1))
            ent = -(alpha.clamp_min(1e-12).log() * alpha).sum(dim=2)
            ent_sum[l] += ent.double().sum(dim=(0, 2, 3, 4))
            ent_sumsq[l] += ent.double().pow(2).sum(dim=(0, 2, 3, 4))
            am = torch.stack(
                [alpha[:, :, idx].sum(dim=2) for idx in axis_idx]
            )  # (3, Lt, H, L, L, L) mass per axis at |Δx| = 1
            aniso = am.std(dim=0, unbiased=False) / am.mean(dim=0).clamp_min(1e-12)
            aniso_sum[l] += aniso.double().mean(dim=(0, 2, 3, 4))
            pick = am.argmax(dim=0)  # (Lt, H, L, L, L) selected axis per site
            for d in range(3):
                axis_counts[l, :, d] += (pick == d).sum(dim=(0, 2, 3, 4)).double()
            if c == 0:
                if field_ell is None:
                    field_ell = torch.zeros(n_layers, H, tg.L, tg.L, tg.L)
                field_ell[l] = ell[0]
        if c == 0:
            field_o = o[0].clone()
        n_samples += tg.LT * tg.L**3

    mean_a = (sum_a / n_samples).float()  # (n_layers, H, n_off) static profile
    std_a = ((sumsq_a / n_samples - (sum_a / n_samples) ** 2).clamp_min(0)).sqrt().float()
    ell = [torch.cat(e) for e in ell_all]  # per layer: (S, H, n_sites)
    o_s = torch.cat(o_all)  # (S, n_sites)
    n_cfg = configs.shape[0]

    # Ensemble-level ℓ_att and its site-to-site spread (the dynamic part).
    ell_mean = torch.stack([e.mean(dim=(0, 2)) for e in ell])  # (n_layers, H)
    ell_std = torch.stack([e.std(dim=(0, 2)) for e in ell])
    ent_mean = (ent_sum / n_samples).float()
    ent_std = ((ent_sumsq / n_samples - (ent_sum / n_samples) ** 2).clamp_min(0)).sqrt().float()
    aniso_mean = (aniso_sum / n_cfg).float()

    # Attention mass per L1 distance (0, 1, 2) — the radial profile.
    dmask = [dist == d for d in range(int(dist.max()) + 1)]
    mass = torch.stack([mean_a[:, :, m].sum(dim=2) for m in dmask], dim=2)

    # ── Ground-truth validation: per-slice Spearman of ℓ(x) against the
    # smeared density o(x), per (layer, head).
    spear_r = torch.zeros(n_layers, H)
    spear_e = torch.zeros(n_layers, H)  # config-level (the quotable one)
    spear_e_slice = torch.zeros(n_layers, H)  # naive slice-level, for the record
    for l in range(n_layers):
        for h in range(H):
            r, e, e_sl = spearman_per_slice(ell[l][:, h], o_s, n_cfg)
            spear_r[l, h], spear_e[l, h], spear_e_slice[l, h] = r, e, e_sl

    # ── Intervention arm: head-ablation matrix. Zeroing w_mix[:, h] in layer l
    # removes head h's contribution to that layer's sublayer output (the
    # residual stream is untouched) — the marginal-value measurement ch. 8
    # requires before any specialization claim.
    obar_base = torch.stack(obars[:N_ABL])
    base = rayleigh_ratios(obar_base)
    base_loss = -sum(base.values()) / len(base)
    abl = torch.zeros(n_layers, H, dtype=torch.float64)
    abl_err = torch.zeros(n_layers, H, dtype=torch.float64)
    for l in tqdm(range(n_layers), desc="ablate"):
        layer = model.gemhsa_models[l]
        for h in range(H):
            orig = layer.w_mix[:, h].detach().clone()
            with torch.no_grad():
                layer.w_mix[:, h] = 0
            ob = []
            for c in range(N_ABL):
                W, T, _ = config_inputs(configs[c])
                O, _ = forward_config(model, W, T, device, collect=False)
                ob.append(O.sum(dim=(1, 2, 3)))
            with torch.no_grad():
                layer.w_mix[:, h] = orig
            abl[l, h], abl_err[l, h] = jackknife_dloss(torch.stack(ob), obar_base)

    # ── Console report ─────────────────────────────────────────────────────────
    print(
        f"\nbaseline Rayleigh loss on {N_ABL} viz configs: {base_loss:.4f} "
        f"(C(1)/C(0) = {base[1]:.4f} → m·a_t ≈ {-np.log(base[1]):.3f})"
    )
    axis_frac = axis_counts / axis_counts.sum(dim=2, keepdim=True)
    print(
        "layer head   ℓ_att        entropy      aniso  axis split      "
        "Spearman(ℓ, o)     Δloss(ablate)"
    )
    for l in range(n_layers):
        for h in range(H):
            sig = abl[l, h] / abl_err[l, h] if abl_err[l, h] > 0 else float("nan")
            print(
                f"  {l + 1}     {h}   {ell_mean[l, h]:.3f}±{ell_std[l, h]:.3f}"
                f"  {ent_mean[l, h]:.3f}±{ent_std[l, h]:.3f}"
                f"  {aniso_mean[l, h]:.3f}"
                f"  {'/'.join(f'{f:.2f}' for f in axis_frac[l, h])}"
                f"  {spear_r[l, h]:+.3f}±{spear_e[l, h]:.3f}"
                f"   {abl[l, h]:+.4f}±{abl_err[l, h]:.4f} ({sig:+.1f}σ)"
            )
    print(
        "\naxis split = fraction of sites selecting each spatial axis at |Δx|=1; "
        "0.33/0.33/0.33 ⇒ content-driven (isotropy restored in the Ō sum),\n"
        "peaked ⇒ a globally fixed axis (spin-2 admixture). "
        "Spearman errors are config-level; the naive slice-level errors would be "
        f"{spear_e_slice.mean():.4f} on average ({(spear_e / spear_e_slice).mean():.1f}× smaller)."
    )

    # ── Figure 1: specialization + validation panels ───────────────────────────
    fig, ax = plt.subplots(2, 3, figsize=(17, 10))
    lcol = [f"C{l}" for l in range(n_layers)]
    hls = ["-", "--", ":", "-."]

    # (0,0) ℓ_att vs layer per head — the RG-coarsening test. Error bars are
    # the site-to-site std: the content-dependent spread, not a mean error.
    for h in range(H):
        ax[0, 0].errorbar(
            np.arange(1, n_layers + 1) + 0.05 * h,
            ell_mean[:, h],
            yerr=ell_std[:, h],
            fmt="o" + hls[h],
            capsize=3,
            label=f"head {h}",
        )
    ax[0, 0].set_xlabel("layer")
    ax[0, 0].set_ylabel(r"$\ell_{\mathrm{att}} = \sum_n \alpha_n |\Delta x_n|_1$")
    ax[0, 0].set_xticks(range(1, n_layers + 1))
    ax[0, 0].set_title("Attention range vs depth\n(bars = site-to-site spread — the content-dependent part)")
    ax[0, 0].legend()
    ax[0, 0].grid(True, alpha=0.3)

    # (0,1) radial attention-mass profile per (layer, head).
    for l in range(n_layers):
        for h in range(H):
            ax[0, 1].plot(
                range(len(dmask)),
                mass[l, h],
                hls[h],
                color=lcol[l],
                marker="o",
                label=f"L{l + 1} h{h}",
            )
    ax[0, 1].set_yscale("log")
    ax[0, 1].set_xticks(range(len(dmask)))
    ax[0, 1].set_xlabel(r"$|\Delta x|_1$")
    ax[0, 1].set_ylabel("mean attention mass at distance")
    ax[0, 1].set_title("Radial profile (config-averaged / static part)")
    ax[0, 1].legend(fontsize=8, ncol=2)
    ax[0, 1].grid(True, alpha=0.3)

    # (0,2) head-ablation matrix — the intervention arm.
    im = ax[0, 2].imshow(abl.numpy(), cmap="Reds", aspect="auto")
    for l in range(n_layers):
        for h in range(H):
            ax[0, 2].text(
                h,
                l,
                f"{abl[l, h]:+.4f}\n±{abl_err[l, h]:.4f}",
                ha="center",
                va="center",
                fontsize=9,
            )
    ax[0, 2].set_xticks(range(H), [f"head {h}" for h in range(H)])
    ax[0, 2].set_yticks(range(n_layers), [f"layer {l + 1}" for l in range(n_layers)])
    ax[0, 2].set_title(
        f"Head ablation: Δ Rayleigh loss (N={N_ABL} configs, delete-one jackknife)\n"
        f"(+ = worse operator; baseline {base_loss:.4f})"
    )
    fig.colorbar(im, ax=ax[0, 2], shrink=0.8)

    # (1,0) distribution of the per-site range — static/dynamic decomposition.
    bins = np.linspace(0, float(dist.max()), 60)
    for l in range(n_layers):
        for h in range(H):
            ax[1, 0].hist(
                ell[l][:, h].reshape(-1).numpy(),
                bins=bins,
                histtype="step",
                density=True,
                color=lcol[l],
                ls=hls[h],
                label=f"L{l + 1} h{h}",
            )
    ax[1, 0].set_xlabel(r"$\ell_{\mathrm{att}}(x)$ per site")
    ax[1, 0].set_ylabel("density")
    ax[1, 0].set_title("Per-site attention range — width = configuration dependence")
    ax[1, 0].legend(fontsize=8, ncol=2)
    ax[1, 0].grid(True, alpha=0.3)

    # (1,1)/(1,2) spatial maps on one z = 0 plane of config 0, t = 0: the
    # (layer, head) with the strongest |Spearman| next to the smeared density.
    lb, hb = np.unravel_index(int(spear_r.abs().argmax()), spear_r.shape)
    f_ell = field_ell[lb, hb, :, :, 0].numpy()
    f_o = field_o[:, :, 0].numpy()
    im1 = ax[1, 1].imshow(f_ell - f_ell.mean(), cmap="coolwarm")
    ax[1, 1].set_title(
        rf"$\delta\ell_{{\mathrm{{att}}}}(x)$  (L{lb + 1} h{hb}, one slice, z=0)"
    )
    fig.colorbar(im1, ax=ax[1, 1], shrink=0.8)
    im2 = ax[1, 2].imshow(f_o - f_o.mean(), cmap="coolwarm")
    ax[1, 2].set_title(
        f"smeared density  δo(x), same slice\n"
        f"Spearman(ℓ, o) = {spear_r[lb, hb]:+.3f} ± {spear_e[lb, hb]:.3f} "
        f"(all {o_s.shape[0]} slices)"
    )
    fig.colorbar(im2, ax=ax[1, 2], shrink=0.8)

    fig.suptitle(
        f"GELT glueball-operator attention as a measurement — Study 3 "
        f"(checkpoint {os.path.basename(CKPT)}, fresh ensemble N={n_cfg}, "
        f"seed {VIZ_SEED})",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig("results/attention/glueball_attention.png", dpi=130, bbox_inches="tight")
    print("Saved results/attention/glueball_attention.png")

    # ── Figure 2: mean-α offset kernels, Δz = 0 plane per (layer, head) ────────
    fig2, ax2 = plt.subplots(H, n_layers, figsize=(4 * n_layers, 4 * H), squeeze=False)
    side = 2 * tg.R + 1
    for h in range(H):
        for l in range(n_layers):
            grid = np.full((side, side), np.nan)
            for n, (dx, dy, dz) in enumerate(offsets):
                if dz == 0:
                    grid[dx + tg.R, dy + tg.R] = mean_a[l, h, n]
            a = ax2[h, l]
            imk = a.imshow(
                np.log10(grid),
                cmap="viridis",
                origin="lower",
                extent=(-tg.R - 0.5, tg.R + 0.5, -tg.R - 0.5, tg.R + 0.5),
            )
            a.set_title(f"layer {l + 1}, head {h}\nself α = {mean_a[l, h, 0]:.3f}")
            a.set_xlabel(r"$\Delta x_1$")
            a.set_ylabel(r"$\Delta x_2$")
            fig2.colorbar(imk, ax=a, shrink=0.8, label=r"$\log_{10}\bar\alpha$")
    fig2.suptitle(
        r"Config-averaged attention kernels $\bar\alpha(\Delta x)$, "
        r"$\Delta x_3 = 0$ plane (static part only)",
        fontsize=13,
    )
    fig2.tight_layout()
    fig2.savefig("results/attention/glueball_attention_kernels.png", dpi=130, bbox_inches="tight")
    print("Saved results/attention/glueball_attention_kernels.png")

    # ── Raw statistics for the thesis figures ──────────────────────────────────
    torch.save(
        {
            "offsets": offsets,
            "mean_alpha": mean_a,
            "std_alpha": std_a,
            "mass_per_distance": mass,
            "ell_mean": ell_mean,
            "ell_std": ell_std,
            "entropy_mean": ent_mean,
            "entropy_std": ent_std,
            "aniso_mean": aniso_mean,
            "axis_counts": axis_counts,
            "axis_frac": axis_frac,
            "spearman_r": spear_r,
            "spearman_err": spear_e,  # config-level — the quotable one
            "spearman_err_slice": spear_e_slice,  # naive, kept for the record
            "ablation_dloss": abl,
            "ablation_err": abl_err,
            "ablation_base_loss": base_loss,
            "meta": {
                "checkpoint": CKPT,
                "viz_seed": VIZ_SEED,
                "n_configs": n_cfg,
                "n_ablation_configs": N_ABL,
                "L": tg.L,
                "Lt": tg.LT,
                "beta": tg.BETA,
                "xi": tg.XI,
                "R": tg.R,
            },
        },
        STATS_DUMP,
    )
    print(f"Saved statistics → {STATS_DUMP}")


if __name__ == "__main__":
    main()
