"""High-statistics held-out evaluation of a TRAINED glueball operator.

The network is frozen (checkpoint selected on its own validation split, exactly
as train_glueball.py does), and the *test* statistics are enlarged by sampling a
fresh, independent SU(2) chain at the same (L, Lt, β, ξ) and running only the
forward pass on it. Nothing is trained or selected on these configurations, so
the result is a sharper measurement of the same net, not a different protocol.
The classical smearing basis is built on the SAME configurations (the ΔA₀
comparison is correlated config by config).

The chain is streamed: 40k configs of 4×24×12³ complex64 links are ~210 GB, so
only chunks live in memory and only Ō(t) is kept. Output has the standard dump
format (`gelt_obar`, `Obar_basis`, `meta`), under the checkpoint's stem plus
`_hs`, so fit_glueball_overlap / su2_fair_fight / two_state_fit read it
unchanged. Resumable: a partial file holds the Ō rows done and the chain state.

Select the net exactly as for train_glueball.py (same env vars), e.g. ens2:

    GLUEBALL_ENSEMBLE_SEED=2 python scripts/high_stat_eval.py

HS_SMOKE=1 HS_DEVICE=cpu HS_BATCH=1 HS_N=4 HS_CHUNK=2 is the plumbing check (any device, untrained net).

Env: HS_N (default 40000), HS_SEED (default 1000 + ensemble seed — must differ
from every seed whose chain trained or validated the net), HS_NSKIP (default
train_glueball.N_SKIP; the jackknife block must then cover 2·τ_int/n_skip
configs — check with block_size_scan.py on the dump), HS_CHUNK (default 60),
HS_TAG (default "_hs").
"""

import functools
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_glueball as tg  # noqa: E402

from gelt.glueball import smearing_operator_basis  # noqa: E402
from gelt.lattice import random_links  # noqa: E402
from gelt.sampler import heatbath_overrelaxation_sweep  # noqa: E402

_bad = [a for a in sys.argv[1:] if a.startswith("--")]
if _bad:
    raise SystemExit(f"unknown flag(s) {_bad}; this script is configured by env (see docstring)")

N = tg._env_int("HS_N", 40000)
SEED = tg._env_int("HS_SEED", 1000 + tg.ENSEMBLE_SEED)
NSKIP = tg._env_int("HS_NSKIP", tg.N_SKIP)
CHUNK = tg._env_int("HS_CHUNK", 60)
TAG = tg._env_str("HS_TAG", "_hs")
# HS_SMOKE: plumbing check on any device — untrained net, no checkpoint needed,
# HS_NTHERM thermalisation sweeps, dump tagged "_smoke" (never a real result).
BATCH = tg._env_int("HS_BATCH", tg.BATCH_CONFIGS)
SMOKE = tg._env_flag("HS_SMOKE", False)
NTHERM = tg._env_int("HS_NTHERM", 2 if SMOKE else tg.N_THERM)
if SMOKE:
    TAG = "_smoke"
if SEED == tg.ENSEMBLE_SEED:
    raise SystemExit("HS_SEED equals the training ensemble seed: the test would not be fresh")

DUMP = tg.CHECKPOINT.replace(".pth", f"{TAG}_test_obars.pt")
PARTIAL = DUMP.replace("_test_obars.pt", "_partial.pt")


def main():
    device = torch.device(
        tg._env_str("HS_DEVICE", "")
        or ("cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu")
    )
    if device.type != "cuda" and not SMOKE:
        raise SystemExit("high_stat_eval refuses to run without CUDA (hours of forward passes)")
    if not SMOKE and not os.path.exists(tg.CHECKPOINT):
        raise SystemExit(f"no checkpoint at {tg.CHECKPOINT}")
    print(f"{tg.NET}  checkpoint {tg.CHECKPOINT}\n→ {DUMP}\nN={N} seed={SEED} n_skip={NSKIP}")

    torch.manual_seed(tg.INIT_SEED)
    model = tg._build_model().to(device)
    if not SMOKE:
        model.load_state_dict(torch.load(tg.CHECKPOINT, map_location=device, weights_only=True))
    model.eval()

    sweep = functools.partial(heatbath_overrelaxation_sweep, n_or=tg.N_OR, xi=tg.XI)
    gaugegroup = tg.gaugegroup

    # chain state: resume from the partial file or thermalise from a Haar start
    g_rows, b_rows, done = [], [], 0
    if os.path.exists(PARTIAL):
        st = torch.load(PARTIAL)
        g_rows, b_rows, done = [st["gelt"]], [st["basis"]], st["done"]
        U = st["U"].to(device)
        print(f"resuming at {done}/{N}")
    else:
        torch.manual_seed(SEED)
        U = random_links(tg.L, tg.D, gaugegroup, dtype=torch.float32, Lt=tg.LT).to(device)
        for _ in tqdm(range(NTHERM), desc="thermalising", leave=False):
            U, _ = sweep(U, gaugegroup, tg.BETA)
    if done:
        torch.manual_seed(SEED + done)  # a fresh stream; still a valid chain

    bar = tqdm(total=N, initial=done, desc="configs")
    while done < N:
        n = min(CHUNK, N - done)
        chunk = []
        for _i in range(n):
            for _j in range(NSKIP):
                U, _acc = sweep(U, gaugegroup, tg.BETA)
            chunk.append(U.clone())
        chunk = torch.stack(chunk).to(tg.MODEL_DTYPE)
        with torch.no_grad():
            g = torch.cat(
                [tg.network_obar(model, chunk[i : i + BATCH], device).cpu()
                 for i in range(0, n, BATCH)]
            ).double()
            b = smearing_operator_basis(
                chunk, gaugegroup, tg.GEVP_LEVELS, alpha=tg.SMEAR_ALPHA
            ).cpu().double()
        g_rows.append(g)
        b_rows.append(b)
        done += n
        bar.update(n)
        torch.save(
            {"gelt": torch.cat(g_rows), "basis": torch.cat(b_rows, dim=1),
             "U": U.cpu(), "done": done},
            PARTIAL,
        )
    bar.close()

    gelt_obar = torch.cat(g_rows)
    Obar_basis = torch.cat(b_rows, dim=1)
    torch.save(
        {
            "gelt_obar": gelt_obar,
            "Obar_basis": Obar_basis,
            "meta": {
                "L": tg.L, "Lt": tg.LT, "beta": tg.BETA, "xi": tg.XI,
                "gevp_levels": list(tg.GEVP_LEVELS), "gevp_t0": tg.GEVP_T0,
                "jack_block": tg.JACK_BLOCK,
                "input_smear_levels": list(tg.INPUT_SMEAR_LEVELS),
                "checkpoint": tg.CHECKPOINT,
                "ensemble_seed": tg.ENSEMBLE_SEED, "init_seed": tg.INIT_SEED,
                "d_model": tg.D_MODEL, "random_init": tg.RANDOM_INIT,
                "arch": tg.ARCH, "lr": tg.LR,
                # a high-statistics dump: fresh chain, frozen net
                "high_stat": True, "hs_seed": SEED, "hs_n_skip": NSKIP, "n_therm": NTHERM,
            },
        },
        DUMP,
    )
    os.remove(PARTIAL)
    print(f"saved {tuple(gelt_obar.shape)} → {DUMP}")


if __name__ == "__main__":
    main()
