"""Raw zero-momentum series for the SU(2) attention-as-operator study — forward only.

``scripts/su2_attention_correlator.py`` measured one trained network against one
random initialisation and analysed the result in place, with the Z₂ run's
estimator (data-chosen window in [2, 8], block 20). For the thesis the study
moves into the 0⁺⁺ chapter, so it has to (i) be read with that chapter's
estimator (fixed window [2, 7], GEVP (t0, td) = (1, 2), block 10), (ii) cover the
three trained networks the headline uses, (iii) put the spread over random
initialisations into the trained − random error, and (iv) answer the question
``notes/thesis_record.md`` §4.2 leaves open: does the attention field carry
ground-state overlap of its own, or does *any* per-site field of a trained net
inherit it from the trained output?

This script does only the GPU half: one pass over the evaluation ensemble, every
network, every field, written as raw (n_ops, B, Nt) series. The statistics are
``scripts/su2_attention_analysis.py``'s, which runs on a laptop from the dump.

Fields per network (all gauge invariant, all summed over the spatial volume):

``out``    ``(B, Nt)``           the network's own operator Ō(t).
``attn``   ``(24, B, Nt)``       the attention field, ordered layer → reduction →
                                 head exactly as ``su2_attention_correlator``.
``resid``  ``(n_layers+1, C, B, Nt)``  Re Tr W_c(x)/N of the residual stream:
                                 stage ℓ < n_layers is the stream *entering*
                                 block ℓ (the features that block's Q and K are
                                 computed from), stage n_layers is the stream
                                 entering the readout. The non-attention control.
``attn_mom``/``resid_mom``       Σ f and Σ f² over sites, for δf/f.

The evaluation ensemble is ``su2_attention_correlator``'s (seed 11, N = 1600,
β = 2.4) and is loaded from its cache when present, so no configuration measured
here was seen by any network.

Run on the V100:

    SAD_SMOKE=1 SAC_DEVICE=cpu python -u scripts/su2_attention_dump.py   # plumbing
    python -u scripts/su2_attention_dump.py

Environment overrides (the SAC_* knobs of su2_attention_correlator also apply:
SAC_N_EVAL, SAC_SEED, SAC_CHUNK, SAC_KEEP, SAC_DEVICE):
    SAD_RANDOM_SEEDS=5   number of random initialisations (seeds RANDOM_SEED + k).
    SAD_SMOKE=1          tiny lattice, random links, random nets only.

Writes ``dumps/su2_attention_series.pt`` (≈ 0.3 GB at the defaults).
"""

import os
import sys
import time

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if os.environ.get("SAD_SMOKE", "0") == "1":
    os.environ["SAC_SMOKE"] = "1"
import su2_attention_correlator as sac  # noqa: E402

tg = sac.tg
SMOKE = sac.SMOKE
N_RANDOM = int(os.environ.get("SAD_RANDOM_SEEDS", 5))
BETA = sac.BETAS[0]
OUT = ("dumps/su2_attention_series_smoke.pt" if SMOKE
       else "dumps/su2_attention_series.pt")

# The three networks behind the chapter-5 headline, one per training ensemble.
# A missing checkpoint is skipped with a warning rather than aborting the run.
TRAINED_CKPTS = {} if SMOKE else {
    f"trained_ens{k}": tg.CHECKPOINT.replace(".pth", "" if k == 0 else f"_ens{k}.pth")
    for k in (0, 1, 2)
}


class ResidualTap:
    """Forward hooks that record Re Tr W_c / N of the residual stream per site.

    One pre-hook per GEMHSA block (the stream entering it) and one on the
    readout's Trace (the stream leaving the stack). Each stage is reduced to
    zero momentum on the fly, so nothing larger than (b·Lt, C) per stage is kept.
    """

    def __init__(self, model):
        self.stages = [None] * (len(model.gemhsa_models) + 1)
        self.handles = []
        for i, layer in enumerate(model.gemhsa_models):
            self.handles.append(layer.register_forward_pre_hook(self._make(i)))
        self.handles.append(
            model.trace.register_forward_pre_hook(self._make(len(model.gemhsa_models)))
        )

    def _make(self, i):
        def hook(_module, args):
            W = args[0]  # (b·Lt, C, L, L, L, nc, nc)
            nc = W.shape[-1]
            f = W.diagonal(dim1=-2, dim2=-1).sum(-1).real / nc  # (b·Lt, C, L, L, L)
            self.stages[i] = (
                f.double().sum(dim=(-3, -2, -1)),  # zero momentum
                f.double().sum(dim=(0, 2, 3, 4)),  # Σ f over sites, per channel
                f.double().pow(2).sum(dim=(0, 2, 3, 4)),  # Σ f²
            )
        return hook


@torch.no_grad()
def measure(model, tap, W, T, b, dist):
    out, attn, attn_mom = sac.attention_fields(model, W, T, b, dist)
    zm = torch.stack([s[0] for s in tap.stages])  # (S, b·Lt, C)
    S, _, C = zm.shape
    resid = zm.reshape(S, b, tg.LT, C).permute(0, 3, 1, 2)  # (S, C, b, Lt)
    resid_mom = torch.stack(
        [torch.stack([s[1], s[2]], dim=-1) for s in tap.stages]
    )  # (S, C, 2)
    return out, attn, attn_mom, resid, resid_mom


def main():
    print(f"device {sac.device} | β = {BETA} | N_EVAL = {sac.N_EVAL} | "
          f"seed {sac.ENSEMBLE_SEED} | {N_RANDOM} random inits")
    nets = {}
    for name, path in TRAINED_CKPTS.items():
        if os.path.exists(path):
            nets[name] = sac.build_model(ckpt=path)
            print(f"  {name}: {path}")
        else:
            print(f"  ** {name}: no checkpoint at {path} — skipped **")
    for k in range(N_RANDOM):
        nets[f"random_s{k}"] = sac.build_model(seed=sac.RANDOM_SEED + k)
    taps = {name: ResidualTap(m) for name, m in nets.items()}
    # Same ℓ = |Δ|₁ table as su2_attention_correlator.main (real dtype: α is a
    # real softmax, a complex weight would promote the reduction).
    offsets = nets["random_s0"].gemhsa_models[0].offsets
    dist = torch.tensor([sum(abs(c) for c in o) for o in offsets],
                        dtype=torch.float32, device=sac.device)

    configs = sac.load_configs(BETA)
    n_cfg = len(configs)
    acc = {name: {"out": [], "attn": [], "resid": [],
                  "attn_mom": 0, "resid_mom": 0} for name in nets}
    classical = []
    t0 = time.time()
    n_chunks = -(-n_cfg // sac.CHUNK)
    for ci, i in enumerate(tqdm(range(0, n_cfg, sac.CHUNK), total=n_chunks)):
        batch = configs[i : i + sac.CHUNK]
        b = len(batch)
        W, T = tg.config_inputs(batch, sac.device)
        classical.append(sac.smearing_operator_basis(
            batch.to(sac.device), sac.gaugegroup, sac.SMEAR_LEVELS,
            alpha=tg.SMEAR_ALPHA).double().cpu())
        for name, model in nets.items():
            out, attn, am, resid, rm = measure(model, taps[name], W, T, b, dist)
            a = acc[name]
            a["out"].append(out.double().cpu())
            a["attn"].append(attn.double().cpu())
            a["resid"].append(resid.cpu())
            a["attn_mom"] = a["attn_mom"] + am.cpu()
            a["resid_mom"] = a["resid_mom"] + rm.cpu()
        del W, T
        if ci == 0:
            dt = time.time() - t0
            print(f"  first chunk {dt:.1f}s → ETA {dt * n_chunks / 60:.1f} min "
                  f"({len(nets)} networks)")

    n_sites = n_cfg * tg.LT * sac.L**3
    blob = {
        "classical": torch.cat(classical, dim=1),  # (n_levels, B, Nt)
        "nets": {
            name: {
                "out": torch.cat(a["out"], dim=0),
                "attn": torch.cat(a["attn"], dim=1),
                "resid": torch.cat(a["resid"], dim=2),
                "attn_mom": a["attn_mom"] / n_sites,
                "resid_mom": a["resid_mom"] / n_sites,
            }
            for name, a in acc.items()
        },
        "meta": {
            "beta": BETA, "N": n_cfg, "Nt": tg.LT, "L": sac.L,
            "ensemble_seed": sac.ENSEMBLE_SEED,
            "smear_levels": list(sac.SMEAR_LEVELS),
            "attn_labels": sac.channel_labels(),
            "resid_stages": [f"in L{l + 1}" for l in range(tg.GEMHSA_LAYERS)]
                            + ["readout"],
            "checkpoints": {n: p for n, p in TRAINED_CKPTS.items() if n in nets},
            "random_seeds": {f"random_s{k}": sac.RANDOM_SEED + k
                             for k in range(N_RANDOM)},
            "smoke": SMOKE,
        },
    }
    os.makedirs("dumps", exist_ok=True)
    torch.save(blob, OUT)
    print(f"wrote {OUT}  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
