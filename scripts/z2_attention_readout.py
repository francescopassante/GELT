"""Is the Z₂ operator's attention structured, or just uniform over the ball?

The go/no-go for the attention-range study, and it costs minutes rather than
the hours another training sweep would.

``ℓ_att`` is bounded by R, but it is also *centred* by the geometry: attention
spread uniformly over the L1 ball has a perfectly well-defined mean range,

    ℓ_uniform = Σ_d (4d · d) / n_off   (2D)   = 4.28 at R = 6,

and the first two β returned ℓ_att = 4.25 and 4.54. Sitting on that number is
exactly what an unstructured map looks like, so a shift in ℓ_att across β
cannot be read as physics until we know the attention is doing anything at
all. The discriminator is the **offset entropy** against its uniform ceiling
log(n_off) = 4.443, plus the radial profile: a structured head concentrates
mass at some scale, a uniform one reproduces the 4d degeneracy of the ball.

This loads the per-β checkpoints ``train_z2_glueball.py`` saved and reports
both, with the uniform references printed beside them. No retraining.

Run (on the box holding the checkpoints and ensembles):
    python scripts/z2_attention_readout.py 0.745 0.752 …

Writes ``datasets/z2_attention_readout.pt``.
"""

import math
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_z2_glueball as tz  # noqa: E402  (path fix must precede the import)

from gelt.blocks_rope import GELT  # noqa: E402


N_VIZ = 16  # held-out configs; attention statistics converge fast
STATS = "datasets/z2_attention_readout.pt"


def build_model():
    return GELT(
        gaugegroup=tz.gaugegroup, L=tz.L, D=2, R=tz.R, nhead=tz.NHEAD,
        gemhsa_layers=tz.GEMHSA_LAYERS, d_qkv=tz.D_QKV, gate=tz.GATE,
        dtype=tz.MODEL_DTYPE, mlp_hidden=tz.MLP_HIDDEN, mlp_out=1,
        reduction="none", init_scale=tz.INIT_SCALE, qk_init_scale=tz.QK_INIT_SCALE,
        mlp_zero_init=False, d_model=tz.D_MODEL, grad_checkpoint=False,
        in_channels=tz.IN_CHANNELS,
    ).to(tz.device)


@torch.no_grad()
def readout(beta):
    tz.BETA = beta
    tz.CACHE = f"datasets/z2_configs_L{tz.L}_Lt{tz.LT}_b{beta}_N{tz.N_CONFIGS}.pt"
    ckpt = f"best_z2_glueball_b{beta}.pth"
    if not os.path.exists(ckpt):
        print(f"  no checkpoint {ckpt} — skipping")
        return None

    # Held-out tail of the chain, matching the training split.
    configs = tz.ensemble()[: tz.N_USE].to(tz.MODEL_DTYPE)
    n_head_cut = int((tz.TRAIN_FRACTION + tz.VAL_FRACTION) * len(configs))
    configs = configs[n_head_cut : n_head_cut + N_VIZ]

    model = build_model()
    model.load_state_dict(torch.load(ckpt, map_location=tz.device, weights_only=True))
    model.eval()

    offsets = model.gemhsa_models[0].offsets
    n_off = len(offsets)
    dist = torch.tensor([sum(abs(c) for c in o) for o in offsets], dtype=torch.float32)
    n_layers, H = tz.GEMHSA_LAYERS, tz.NHEAD

    sum_a = torch.zeros(n_layers, H, n_off, dtype=torch.float64)
    ent_sum = torch.zeros(n_layers, H, dtype=torch.float64)
    ell_sum = torch.zeros(n_layers, H, dtype=torch.float64)
    n_samples = 0

    for c in tqdm(range(len(configs)), desc=f"β={beta}", leave=False):
        W, T = tz.config_inputs(configs[c : c + 1])
        model(W, T)
        for l, layer in enumerate(model.gemhsa_models):
            a = layer._last_alpha.cpu().double()  # (b·Lt, H, n_off, L, L)
            flat = a.permute(1, 2, 0, 3, 4).reshape(H, n_off, -1)  # (H, n_off, S)
            sum_a[l] += flat.sum(dim=2)
            ell_sum[l] += (flat * dist.double().view(1, n_off, 1)).sum(dim=(1, 2))
            ent_sum[l] += -(flat.clamp_min(1e-12).log() * flat).sum(dim=(1, 2))
            if l == 0:
                n_samples += flat.shape[2]

    mean_a = (sum_a / n_samples).float()
    ell = (ell_sum / n_samples).float()
    ent = (ent_sum / n_samples).float()

    # Uniform references — the numbers the measurement must beat to mean anything.
    ell_unif = float((dist.double().sum() / n_off).item())
    ent_unif = math.log(n_off)

    print(f"\nβ = {beta}   (n_off = {n_off}, R = {tz.R})")
    print(f"  uniform references:  ℓ_att = {ell_unif:.3f}   entropy = {ent_unif:.3f}")
    print("  layer head   ℓ_att     entropy   ent/uniform   self-α")
    for l in range(n_layers):
        for h in range(H):
            print(f"    {l + 1}     {h}   {ell[l, h]:>6.3f}    {ent[l, h]:>6.3f}"
                  f"      {ent[l, h] / ent_unif:>6.3f}     {mean_a[l, h, 0]:.4f}")

    # Radial profile against the uniform expectation 4d/n_off.
    dmax = int(dist.max())
    print("  radial mass (mean over heads) vs uniform:")
    for d in range(dmax + 1):
        m = dist == d
        meas = mean_a[:, :, m].sum(dim=2).mean().item()
        unif = int(m.sum()) / n_off
        print(f"    |Δx|={d}:  {meas:.4f}   uniform {unif:.4f}   ratio {meas / unif:.2f}")

    worst = (ent / ent_unif).max().item()
    print(f"  VERDICT: max entropy/uniform = {worst:.3f}"
          + ("  → essentially UNIFORM; ℓ_att is measuring the ball, not the physics"
             if worst > 0.97 else
             "  → structured; the ℓ_att trend is worth reading"))
    return {
        "beta": beta, "mean_alpha": mean_a, "ell": ell, "entropy": ent,
        "ell_uniform": ell_unif, "entropy_uniform": ent_unif, "n_off": n_off,
    }


def main():
    betas = [float(a) for a in sys.argv[1:]] or [0.745, 0.752]
    out = {}
    for b in betas:
        r = readout(b)
        if r is not None:
            out[b] = r
    if out:
        torch.save(out, STATS)
        print(f"\nSaved {STATS}")


if __name__ == "__main__":
    main()
