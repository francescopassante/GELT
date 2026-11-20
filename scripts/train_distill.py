"""Transfer-matrix distillation: a single-timeslice operator trained to predict
the Monte Carlo's own Euclidean future. V100; one training ≈ one train_glueball run.

The identity it rests on. Let y be any gauge-invariant observable supported on
timeslices other than t, and U_t the spatial links of slice t. The spatial links
of one slice are the Markov state of the transfer matrix T, so with ψ₀ the
vacuum wavefunctional and E₀ = 0,

    g*(U) := E[ y(t + a) | U_t = U ] = (T^a ŷ ψ₀)(U) / ψ₀(U),

i.e. the *multiplication* operator ĝ* acts on the vacuum as  ĝ*|0⟩ = T^a ŷ|0⟩.
The regression function of the future on the present is the Euclidean-evolved
operator, as a legitimate single-timeslice operator: its correlator is
C_g*(Δ) = C_y(Δ + 2a) and every excited state in it is suppressed by
e^{−2(E_n−E_1)a} relative to y — the filter a multilevel update buys with nested
Monte Carlo, here amortised into a function fitted once on the training
ensemble. By time reflection E[y(t − a) | U_t] is the same function, so both
directions are targets. Linearity makes it hold site by site: with o(x, t) any
local density of y, E[o(x, t ± a) | U_t] is a local density of g*.

Whatever the student ends up being, it is a fixed function of one timeslice
evaluated on configurations it never saw, so its correlator has the usual
positive spectral decomposition and the usual variational bound: an imperfect
regression costs overlap, never correctness. In a *linear* span the same
regression is one power-iteration step of C(0)⁻¹C(a), and its ceiling is
scripts/variational_headroom.py's η_S(a); this script asks whether a network
reaches the part of T^a|y⟩ that no span in the repo expresses (1 − η ≈ 3% per
timeslice, ≈ 12% of the state in all — see that script's tables).

What it runs. Teacher = a trained single-operator net (frozen); student = the
same architecture, started from the teacher's weights (TMD_INIT=teacher) so the
function class is held fixed and only the training principle changes. Per batch
both see the identical (W, T); the target is the teacher's own per-site field at
the other timeslices of the same configuration,

    F(x, t) = mean_{a ∈ LAGS} ½ [ y_T(x, t + a) + y_T(x, t − a) ],

with y = (o − μ)/σ in the teacher's units (μ, σ fixed once from the first
TMD_NORM_CONFIGS training configurations). Loss: mean squared error of the
fields averaged over spatial blocks of side TMD_POOL (a low-pass: the predictable
part of the future lives at low momentum, the per-site remainder is irreducible
noise), plus TMD_P0_WEIGHT × the zero-momentum error. TMD_POOL = L is the
zero-momentum regression alone.

Connected loss (TMD_LOSS=connected, the default since 2026-11-20). The error is
centred over the minibatch (6 configurations × Lt slices × V sites) before both
terms, so the loss does not see a constant offset — which no connected
correlator sees either. The first run (ens2, lag 2, `_tmd2`, the raw loss) showed
why it must not: the teacher's per-site field has μ/σ ≈ −55, so a rescale of the
raw output by c moves the mean by 55·(1 − c)σ unless the last bias compensates
exactly, and the zero-momentum term weighs an offset b as V·b² = 1728·b². Its
validation loss swung 6 → 37 → 6 between epochs on offsets of ~0.12σ (the block
and zero-momentum excesses were in the ratio V/POOL³ = 64, the signature of a
constant), against a purification signal of 0.50 out of 6.4.
TMD_LOSS=raw is the loss of that run, kept to reproduce it.

Selection: under the connected loss, the lowest validation *shape* loss — the
loss after the best rescale c* of the student's field, fitted in closed form on
the validation split (min_c Q(c·s − t) = Q(t) − B(s,t)²/Q(s), Q the loss's
quadratic form and B its bilinear form). The rescale is a legitimate part of the
regression (c* ≈ C(2a)/C(0) for a student that is still its teacher) but no
readout sees it, and in the first run it was the whole of the loss decrease:
student 6.51 on test, the teacher rescaled 6.43, the exact regression function
≈ 5.94. Under TMD_LOSS=raw, the raw validation loss, as in that run. Never a
correlator of the student, never the test split. Splits, ensemble cache, inputs, classical basis
and dump format are train_glueball's by import, so every offline script reads
the student's dump unchanged; the dump also carries the teacher's test series
(`teacher_obar`), on the same configurations, for the paired readout
(scripts/tmd_readout.py, where the readings are fixed).

The student's checkpoint is named as train_glueball would name a run tagged
`<GLUEBALL_RUN_TAG>_tmd<lags><TMD_TAG>`, so e.g.

    GLUEBALL_ENSEMBLE_SEED=2 GLUEBALL_RUN_TAG=_tmd2 \\
        HS_ALSO=results/glueball/best_glueball_gelt_sm0-2-4-6_ens2.pth \\
        python scripts/high_stat_eval.py

evaluates it (and its teacher, on the same fresh chain) at high statistics.

Env, each also as --tmd-name=value (the architecture is selected exactly as for
train_glueball.py, by the same GLUEBALL_* variables or flags):
  TMD_LAGS         comma list of lags a, default "2"
  TMD_TEACHER      teacher checkpoint (default: the untagged run of this stem)
  TMD_TAG          extra suffix, e.g. "_r2" for a second round whose teacher is
                   the first round's student (must start with "_")
  TMD_INIT         "teacher" (default) or "scratch" (train_glueball's init)
  TMD_LOSS         "connected" (default: offset-free loss, selection on the
                   shape loss) or "raw" (the first run's loss and selection).
                   Connected students carry "_c" in their tag: _tmd2_c.
  TMD_POOL         spatial block side of the loss, default 3 (must divide L)
  TMD_P0_WEIGHT    weight of the zero-momentum error, default 1.0
  TMD_LR / TMD_EPOCHS / TMD_PATIENCE   default 3e-4 / 20 / 6
  TMD_WARMUP       optimiser steps of linear warm-up, default 100. The student
                   starts from converged weights and Adam's first steps have
                   size lr whatever the gradient: without the ramp one step at
                   1e-3 took the plumbing check's loss from 0.5 to 138.
  TMD_NORM_CONFIGS default 12
  TMD_EVAL_ONLY=1  skip training, redo the evaluation of the saved student
  TMD_SMOKE=1      plumbing check on any device: a 4 × 6³ Haar ensemble, an
                   untrained teacher, two tiny epochs, artifacts under
                   TMD_SMOKE_DIR (default results/glueball/smoke). Never a result.

Run:  GLUEBALL_ENSEMBLE_SEED=2 python scripts/train_distill.py
"""

import os
import sys

# Every variable is also accepted as --name=value (TMD_LAGS ↔ --tmd-lags=2), as
# in train_glueball.py: env vars do not survive every container wrapper. An
# unrecognised --tmd-* flag is refused rather than ignored.
_TMD = ("lags", "teacher", "tag", "init", "loss", "pool", "p0-weight", "lr", "epochs", "patience",
        "warmup", "norm-configs", "eval-only", "smoke", "smoke-dir", "device")
_bad = [a for a in sys.argv[1:] if a.startswith("--tmd-")
        and a[len("--tmd-"):].split("=", 1)[0] not in _TMD]
if _bad:
    raise SystemExit(f"unknown flag(s) {_bad}; known: {['--tmd-' + k for k in _TMD]}")


def _arg(name, default):
    """``TMD_FOO`` from ``--tmd-foo=value`` in argv, else from the environment."""
    flag = "--" + name.lower().replace("_", "-") + "="
    for a in sys.argv[1:]:
        if a.startswith(flag):
            return a.split("=", 1)[1]
    return os.environ.get(name, default)


LAGS = tuple(int(x) for x in _arg("TMD_LAGS", "2").split(","))
TMD_TAG = _arg("TMD_TAG", "")
LOSS = _arg("TMD_LOSS", "connected").lower()
if LOSS not in ("connected", "raw"):
    raise SystemExit(f"TMD_LOSS must be 'connected' or 'raw' (got {LOSS!r})")
SMOKE = _arg("TMD_SMOKE", "0") not in ("0", "false", "False", "")
if TMD_TAG and not TMD_TAG.startswith("_"):
    raise SystemExit(f"TMD_TAG must start with '_' (got {TMD_TAG!r})")
if not LAGS or min(LAGS) < 1:
    raise SystemExit(f"TMD_LAGS must be positive integers (got {LAGS})")
# The student is named as a train_glueball run carrying this tag, so that every
# script that selects a net by env (high_stat_eval, cubic_projection) finds it.
# train_glueball reads --run-tag= from argv before the environment, so a tag
# given that way is folded in here and withdrawn from argv.
_BASE_TAG = _arg("GLUEBALL_RUN_TAG", "") if not any(
    a.startswith("--run-tag=") for a in sys.argv[1:]) else next(
    a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--run-tag="))
sys.argv = [a for a in sys.argv if not a.startswith("--run-tag=")]
# A connected-loss student is "_c"-suffixed, so it can never overwrite the raw
# run's checkpoint (results/glueball/…_ens2_tmd2.pth is the recorded R-T0).
_STUDENT_TAG = (_BASE_TAG + "_tmd" + "-".join(str(a) for a in LAGS)
                + ("_c" if LOSS == "connected" else "") + TMD_TAG)
os.environ["GLUEBALL_RUN_TAG"] = _STUDENT_TAG

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F_  # noqa: E402
from tqdm import tqdm  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_glueball as tg  # noqa: E402

from gelt.glueball import smearing_operator_basis  # noqa: E402
from gelt.lattice import random_links  # noqa: E402

if tg.N_STATES != 1:
    raise SystemExit("train_distill distils one operator; unset GLUEBALL_N_STATES")
if tg.RANDOM_INIT or tg.WARM_FROM:
    raise SystemExit("GLUEBALL_RANDOM_INIT / GLUEBALL_WARM_FROM do not apply here; "
                     "the student's start is TMD_INIT")

INIT = _arg("TMD_INIT", "teacher").lower()
if INIT not in ("teacher", "scratch"):
    raise SystemExit(f"TMD_INIT must be 'teacher' or 'scratch' (got {INIT!r})")
POOL = int(_arg("TMD_POOL", 3))
P0_WEIGHT = float(_arg("TMD_P0_WEIGHT", 1.0))
LR = float(_arg("TMD_LR", 3e-4))
WARMUP = int(_arg("TMD_WARMUP", 100))  # optimiser steps of linear warm-up
EPOCHS = int(_arg("TMD_EPOCHS", 20))
PATIENCE = int(_arg("TMD_PATIENCE", 6))
NORM_CONFIGS = int(_arg("TMD_NORM_CONFIGS", 12))
EVAL_ONLY = str(_arg("TMD_EVAL_ONLY", "0")) not in ("0", "false", "False", "")

if SMOKE:
    # A lattice small enough for a CPU, through the very same code path. The
    # module globals are what config_inputs / _build_model read at call time.
    tg.L, tg.LT, tg.BATCH_CONFIGS, tg.JACK_BLOCK = 6, 4, 2, 2
    EPOCHS, NORM_CONFIGS = min(EPOCHS, 2), 2
    _dir = _arg("TMD_SMOKE_DIR", "results/glueball/smoke")
    os.makedirs(_dir, exist_ok=True)
    STUDENT = os.path.join(_dir, os.path.basename(tg.CHECKPOINT))
else:
    STUDENT = tg.CHECKPOINT
_suffix = _STUDENT_TAG + ".pth"
assert STUDENT.endswith(_suffix), (STUDENT, _suffix)
TEACHER = _arg("TMD_TEACHER", "") or (
    tg.CHECKPOINT[: -len(_suffix)] + _BASE_TAG + ".pth")
DUMP = STUDENT.replace(".pth", "_test_obars.pt")
if tg.L % POOL:
    raise SystemExit(f"TMD_POOL = {POOL} must divide L = {tg.L}")


# ── the objective ─────────────────────────────────────────────────────────────
def lagged_target(y):
    """F(x, t) = mean over LAGS of ½[y(x, t+a) + y(x, t−a)].  ``y`` : (b, Lt, *Λ)."""
    return sum(0.5 * (y.roll(-a, 1) + y.roll(a, 1)) for a in LAGS) / len(LAGS)


def tmd_loss(ys, target):
    """Block-averaged field error + P0_WEIGHT × zero-momentum error.

    ``ys``, ``target`` : (b, Lt, L, L, L) in the teacher's units. Both terms are
    normalised so that a spatially white error of variance s² scores s²: the
    block mean of POOL³ sites has variance s²/POOL³, the slice mean s²/V.
    Under TMD_LOSS=connected the error is first centred over the whole batch,
    so a constant offset costs nothing (the module docstring says why).
    Returns ``(loss, block term, zero-momentum term)``.
    """
    e = ys - target
    if LOSS == "connected":
        e = e - e.mean()
    b, Lt = e.shape[:2]
    sp = e.shape[2:]
    if POOL > 1:
        eb = F_.avg_pool3d(e.reshape(b * Lt, 1, *sp), POOL)
        block = eb.pow(2).mean() * POOL ** 3
    else:
        block = e.pow(2).mean()
    zero = e.mean(dim=(2, 3, 4)).pow(2).mean() * float(np.prod(sp))
    return block + P0_WEIGHT * zero, block, zero


def _loss_features(x):
    """``x`` → (block field, zero-momentum field) such that the loss's quadratic
    form is Q(x) = mean(block²) + P0_WEIGHT·mean(zero²) — the same normalisation
    as ``tmd_loss`` (centred first under the connected loss)."""
    if LOSS == "connected":
        x = x - x.mean()
    b, Lt = x.shape[:2]
    sp = x.shape[2:]
    xb = (F_.avg_pool3d(x.reshape(b * Lt, 1, *sp), POOL) * POOL ** 1.5 if POOL > 1 else x)
    xz = x.mean(dim=(2, 3, 4)) * float(np.prod(sp)) ** 0.5
    return xb, xz


def quad_stats(s, t):
    """[Q(s), B(s, t), Q(t)] of one batch, float64: everything ``shape_loss`` needs."""
    (sb, sz), (tb, tz) = _loss_features(s.double()), _loss_features(t.double())
    return torch.stack([
        (sb * sb).mean() + P0_WEIGHT * (sz * sz).mean(),
        (sb * tb).mean() + P0_WEIGHT * (sz * tz).mean(),
        (tb * tb).mean() + P0_WEIGHT * (tz * tz).mean(),
    ]).cpu()


def shape_loss(q):
    """(min_c Q(c·s − t), c*) from summed ``quad_stats`` — the loss after the best
    rescale of the student, which is what a scale-invariant readout can see."""
    c = (q[1] / q[0]).item()
    return (q[2] - q[1] ** 2 / q[0]).item(), c


def site_fields(model, W, T, b):
    """The per-site output of ``model`` → (b, Lt, L, L, L), real."""
    o = model(W, T)
    return o.view(b, o.shape[0] // b, *o.shape[1:])


@torch.no_grad()
def evaluate(student, teacher, configs, mu, sigma, device):
    """One pass: the distillation loss, the shape loss and both zero-momentum series.

    Returns a dict: ``loss``, ``block``, ``zero`` (means over batches weighted by
    their size), ``shape`` and ``c`` (the student after its best rescale),
    ``shape_teacher`` and ``c_teacher`` (the same for the teacher's own field —
    the start's value, on the same configurations), ``s_obar`` and ``t_obar``
    (N, Lt).
    """
    student.eval()
    tot, n, so, to = torch.zeros(3, dtype=torch.float64), 0, [], []
    qs, qt = torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64)
    for i in range(0, configs.shape[0], tg.BATCH_CONFIGS):
        batch = configs[i: i + tg.BATCH_CONFIGS]
        b = batch.shape[0]
        W, T = tg.config_inputs(batch, device)
        oT = site_fields(teacher, W, T, b)
        oS = site_fields(student, W, T, b)
        yT = (oT - mu) / sigma
        yS = (oS - mu) / sigma
        target = lagged_target(yT)
        terms = tmd_loss(yS, target)
        tot += torch.stack([t.double().cpu() for t in terms]) * b
        qs += quad_stats(yS, target) * b
        qt += quad_stats(yT, target) * b
        n += b
        so.append(oS.sum(dim=(2, 3, 4)).cpu())
        to.append(oT.sum(dim=(2, 3, 4)).cpu())
    loss, block, zero = (tot / n).tolist()
    shape, c = shape_loss(qs / n)
    shape_t, c_t = shape_loss(qt / n)
    return {"loss": loss, "block": block, "zero": zero, "shape": shape, "c": c,
            "shape_teacher": shape_t, "c_teacher": c_t,
            "s_obar": torch.cat(so), "t_obar": torch.cat(to)}


def selection_score(v):
    """What selection minimises: the shape loss under the connected loss, the raw
    validation loss under TMD_LOSS=raw (the first run's criterion)."""
    return v["shape"] if LOSS == "connected" else v["loss"]


def rayleigh_summary(obar):
    """(C(1)/C(0), C(2)/C(0)) of a series, with the estimator the training loss uses."""
    d = obar.double() - obar.double().mean()
    C = [(d * d.roll(-k, 1)).mean().item() for k in range(3)]
    return C[1] / C[0], C[2] / C[0]


def main():
    device = torch.device(
        _arg("TMD_DEVICE", "")
        or ("cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"))
    if device.type != "cuda" and not SMOKE:
        raise SystemExit("train_distill refuses to run without CUDA (TMD_SMOKE=1 is the "
                         "plumbing check)")
    print(f"device {device} | lags {LAGS} | loss {LOSS} | pool {POOL} | p0 weight {P0_WEIGHT:g} | "
          f"init {INIT} | lr {LR:g} (warm-up {WARMUP} steps) | epochs {EPOCHS}\nteacher {TEACHER}\nstudent {STUDENT}")

    # ── ensemble: train_glueball's cache, never sampled here ──────────────────
    if SMOKE:
        torch.manual_seed(0)
        configs = torch.stack([random_links(tg.L, tg.D, tg.gaugegroup, Lt=tg.LT)
                               for _ in range(10)])
    else:
        if not os.path.exists(tg.CACHE):
            raise SystemExit(f"no cached ensemble at {tg.CACHE}: this script samples nothing "
                             f"(train_glueball.py writes the cache)")
        if not os.path.exists(TEACHER):
            raise SystemExit(f"no teacher checkpoint at {TEACHER} (set TMD_TEACHER)")
        configs = torch.load(tg.CACHE)
    configs = configs.to(tg.MODEL_DTYPE)
    N = configs.shape[0]
    n_train = int(round(tg.TRAIN_FRACTION * N))
    n_val = int(round(tg.VAL_FRACTION * N))
    train_configs = configs[:n_train]
    val_configs = configs[n_train: n_train + n_val]
    test_configs = configs[n_train + n_val:]
    print(f"split (contiguous): {n_train} train / {n_val} val / {test_configs.shape[0]} test")

    # ── teacher (frozen) and student ──────────────────────────────────────────
    torch.manual_seed(tg.INIT_SEED)
    teacher = tg._build_model().to(device)
    if not SMOKE:
        teacher.load_state_dict(torch.load(TEACHER, map_location=device, weights_only=True))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    torch.manual_seed(tg.INIT_SEED + 1)
    student = tg._build_model().to(device)
    if INIT == "teacher":
        student.load_state_dict(teacher.state_dict())

    # ── the teacher's units: μ, σ of its per-site field, fixed once ───────────
    with torch.no_grad():
        acc = []
        for i in range(0, min(NORM_CONFIGS, n_train), tg.BATCH_CONFIGS):
            batch = train_configs[i: i + tg.BATCH_CONFIGS]
            W, T = tg.config_inputs(batch, device)
            acc.append(site_fields(teacher, W, T, batch.shape[0]).double().flatten().cpu())
        acc = torch.cat(acc)
        mu, sigma = acc.mean().item(), acc.std().item()
    if not (sigma > 0 and np.isfinite(sigma)):
        raise SystemExit(f"teacher field has σ = {sigma}: nothing to regress on")
    print(f"teacher field: μ = {mu:.6g}, σ = {sigma:.6g} (first {NORM_CONFIGS} train configs)")

    history = {"train": [], "val": [], "val_block": [], "val_zero": [], "val_shape": [],
               "val_c": [], "val_R1": [], "val_R2": []}
    v0 = evaluate(student, teacher, val_configs, mu, sigma, device)
    tR = rayleigh_summary(v0["t_obar"])
    sR = rayleigh_summary(v0["s_obar"])
    print(f"start: val loss {v0['loss']:.5f} (block {v0['block']:.5f}, zero {v0['zero']:.5f}) | "
          f"shape {v0['shape']:.5f} at c* {v0['c']:.3f} | "
          f"C(1)/C(0), C(2)/C(0): teacher {tR[0]:.4f}, {tR[1]:.4f}  student {sR[0]:.4f}, "
          f"{sR[1]:.4f}")
    best_val = float("inf")

    if not EVAL_ONLY:
        if INIT == "teacher":
            # The start is a candidate: a student that never improves on its own
            # starting validation loss must be reported as its teacher (the null
            # of tmd_readout.py), not as whatever the first epoch left behind.
            best_val = selection_score(v0)
            torch.save(student.state_dict(), STUDENT)
        opt = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=tg.WEIGHT_DECAY)
        steps_per_epoch = 2 if SMOKE else -(-n_train // tg.BATCH_CONFIGS)
        total = max(1, EPOCHS * steps_per_epoch)
        # Linear warm-up into one cosine decay, stepped per optimiser step.
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda k: min(1.0, (k + 1) / max(1, WARMUP))
            * 0.5 * (1.0 + np.cos(np.pi * min(k, total) / total)))
        torch.manual_seed(tg.INIT_SEED)
        stale = 0
        try:
            for epoch in range(EPOCHS):
                student.train()
                order = torch.randperm(n_train)
                run, nb, skipped = 0.0, 0, 0
                steps = range(0, n_train, tg.BATCH_CONFIGS)
                if SMOKE:
                    steps = list(steps)[:2]
                bar = tqdm(steps, desc=f"epoch {epoch + 1}", leave=False)
                for i in bar:
                    batch = train_configs[order[i: i + tg.BATCH_CONFIGS]]
                    b = batch.shape[0]
                    W, T = tg.config_inputs(batch, device)
                    with torch.no_grad():
                        target = lagged_target(
                            (site_fields(teacher, W, T, b) - mu) / sigma)
                    opt.zero_grad()
                    loss, _, _ = tmd_loss((site_fields(student, W, T, b) - mu) / sigma, target)
                    if not torch.isfinite(loss):
                        skipped += 1
                        continue
                    loss.backward()
                    gn = torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                    if not torch.isfinite(gn):
                        skipped += 1
                        continue
                    opt.step()
                    sched.step()
                    run += loss.item()
                    nb += 1
                    bar.set_postfix(loss=f"{loss.item():.5f}")
                if nb == 0:
                    raise SystemExit(f"epoch {epoch + 1}: every batch was non-finite")
                v = evaluate(student, teacher, val_configs, mu, sigma, device)
                R1, R2 = rayleigh_summary(v["s_obar"])
                for k, x in zip(history, (run / nb, v["loss"], v["block"], v["zero"],
                                          v["shape"], v["c"], R1, R2)):
                    history[k].append(x)
                mark = ""
                if selection_score(v) < best_val:
                    best_val, stale, mark = selection_score(v), 0, "  *"
                    torch.save(student.state_dict(), STUDENT)
                else:
                    stale += 1
                print(f"epoch {epoch + 1:3d}: train {run / nb:.5f} | val {v['loss']:.5f} "
                      f"(block {v['block']:.5f}, zero {v['zero']:.5f}) | shape "
                      f"{v['shape']:.5f} at c* {v['c']:.3f} | student val C(1)/C(0) "
                      f"{R1:.4f}, C(2)/C(0) {R2:.4f}"
                      + (f" | skipped {skipped}" if skipped else "") + mark)
                if stale >= PATIENCE:
                    print(f"no val improvement in {PATIENCE} epochs — stopping")
                    break
        except KeyboardInterrupt:
            print("\ninterrupted — evaluating the best checkpoint so far")

    if not os.path.exists(STUDENT):
        raise SystemExit(f"no student checkpoint at {STUDENT}: nothing to evaluate")
    student.load_state_dict(torch.load(STUDENT, map_location=device, weights_only=True))

    # ── test split: untouched until here ──────────────────────────────────────
    t = evaluate(student, teacher, test_configs, mu, sigma, device)
    s_obar, t_obar = t["s_obar"].double(), t["t_obar"].double()
    print(f"\ntest distillation loss {t['loss']:.5f} (block {t['block']:.5f}, "
          f"zero {t['zero']:.5f})")
    print(f"test shape loss (after the best rescale): student {t['shape']:.5f} at c* "
          f"{t['c']:.3f} | teacher {t['shape_teacher']:.5f} at c* {t['c_teacher']:.3f} | "
          f"Δ {t['shape'] - t['shape_teacher']:+.5f}")
    for lbl, o in (("teacher", t_obar), ("student", s_obar)):
        m, e = tg.blocked_jackknife_meff(o, tg.JACK_BLOCK)
        R1, R2 = rayleigh_summary(o)
        print(f"  {lbl}: C(1)/C(0) {R1:.4f}  C(2)/C(0) {R2:.4f}  m_eff(Δ=0..4) "
              + "  ".join(f"{m[k].item():.3f}({e[k].item() * 1e3:.0f})"
                          for k in range(min(5, m.shape[0]))))
    print("  ideal student: m_eff^S(Δ) = m_eff^T(Δ + 2a). The paired readings are "
          "scripts/tmd_readout.py's.")

    Obar_basis = smearing_operator_basis(
        test_configs, tg.gaugegroup, tg.GEVP_LEVELS, alpha=tg.SMEAR_ALPHA
    ).cpu().double()
    torch.save(
        {
            "gelt_obar": s_obar,
            "teacher_obar": t_obar,
            "Obar_basis": Obar_basis,
            "meta": {
                "L": tg.L, "Lt": tg.LT, "beta": tg.BETA, "xi": tg.XI,
                "gevp_levels": list(tg.GEVP_LEVELS), "gevp_t0": tg.GEVP_T0,
                "jack_block": tg.JACK_BLOCK,
                "input_smear_levels": list(tg.INPUT_SMEAR_LEVELS),
                "checkpoint": STUDENT, "best_val_loss": best_val,
                "ensemble_seed": tg.ENSEMBLE_SEED, "init_seed": tg.INIT_SEED,
                "d_model": tg.D_MODEL, "d_qkv": tg.D_QKV, "alpha_mode": tg.ALPHA_MODE,
                "random_init": False, "arch": tg.ARCH, "lr": LR, "n_states": 1,
                # what makes this a distilled operator and not a Rayleigh-trained one
                "tmd": {"lags": list(LAGS), "teacher": TEACHER, "init": INIT, "loss": LOSS,
                        "pool": POOL,
                        "p0_weight": P0_WEIGHT, "mu": mu, "sigma": sigma,
                        "epochs": EPOCHS, "warmup": WARMUP, "history": history,
                        "val_loss_start": v0["loss"], "val_shape_start": v0["shape"],
                        "test_loss": t["loss"], "test_shape": t["shape"], "test_c": t["c"],
                        "test_shape_teacher": t["shape_teacher"],
                        "test_c_teacher": t["c_teacher"], "smoke": SMOKE},
            },
        },
        DUMP,
    )
    print(f"saved → {DUMP}")


if __name__ == "__main__":
    main()
