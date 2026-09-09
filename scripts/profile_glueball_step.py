"""Timing breakdown of one train_glueball.py training step.

Answers "where does the step time go?" — over the pipeline the training loop
actually runs. It reuses ``train_glueball``'s constants, ``config_inputs``,
model construction and loss, so the numbers reflect the real step rather than a
paraphrase of it.

The previous version of this script built its inputs inline: one thin
``plaquette_tensor`` level and no APE ladder. That understated the input stage
by the whole smearing cost (``INPUT_SMEAR_LEVELS = (0, 2, 4, 6)`` is six
smearing iterations *per optimizer step*) and is where the "transport is 1.9%"
figure in CLAUDE.md comes from — true of the transport, but measured against a
denominator that was missing a stage. Everything here goes through
``tg.config_inputs``.

Uses the cached ensemble if present, otherwise Haar-random configs (identical
compute cost — the timings don't depend on the link values).

Run (from the repo root):
    python scripts/profile_glueball_step.py [batch_configs]

``batch_configs`` defaults to train_glueball.BATCH_CONFIGS; pass e.g. 12 to
probe a bigger batch's step time/memory, or 2 on a small local GPU.

Env switches (each prints its own comparison line, so one run can A/B them):
    PROFILE_DIAGNOSTICS=1  also time the step with the per-layer introspection
                           stashes on — the pre-2026-09 default. Each block
                           reduces two offset-expanded tensors and calls
                           .item() ~10 times, i.e. ~10 GPU syncs per block per
                           forward, and again inside every checkpoint recompute.
    PROFILE_LEGACY_SMEAR=1 also time the APE ladder the way it used to be built
                           (Python loop over configs, SU.project via batched
                           2×2 SVD + det) against the current vectorised,
                           closed-form route.
    PROFILE_MICRO=0        skip the per-stage forward/backward micro-benchmark
                           (on by default; it allocates one layer's worth of
                           offset-expanded tensors, so drop it if memory is
                           tight).
    PROFILE_COMPILE=1      also time the step with torch.compile on the model.
                           Unproven on complex64 — this is here to get a number,
                           not because it is expected to work.
"""

import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_glueball as tg

from gelt.blocks import GELT
from gelt.glueball import ape_smear
from gelt.lattice import build_transport_average, plaquette_tensor, random_links
from gelt.sampler import staple_sum

N_WARMUP = 2  # untimed iterations (JIT/cudnn autotune, allocator warm-up)
N_TIMED = 5


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _device():
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def _batch(b, device):
    if os.path.exists(tg.CACHE):
        print(f"using first {b} configs of cached ensemble {tg.CACHE}")
        return torch.load(tg.CACHE)[:b].to(tg.MODEL_DTYPE)
    print(f"cache absent — using {b} Haar-random configs (same compute cost)")
    return torch.stack(
        [random_links(tg.L, tg.D, tg.gaugegroup, Lt=tg.LT) for _ in range(b)]
    ).to(tg.MODEL_DTYPE)


def _model(device):
    return GELT(
        gaugegroup=tg.gaugegroup, L=tg.L, D=3, R=tg.R, nhead=tg.NHEAD,
        gemhsa_layers=tg.GEMHSA_LAYERS, d_qkv=tg.D_QKV, gate=tg.GATE,
        dtype=tg.MODEL_DTYPE, mlp_hidden=tg.MLP_HIDDEN, mlp_out=1,
        reduction="none", init_scale=tg.INIT_SCALE, qk_init_scale=tg.QK_INIT_SCALE,
        mlp_zero_init=False, d_model=tg.D_MODEL,
        grad_checkpoint=tg.GRAD_CHECKPOINT,
        in_channels=3 * len(tg.INPUT_SMEAR_LEVELS),
    ).to(device)


def time_step(model, batch, device, label, compile_model=False):
    """Per-stage means over N_TIMED real training steps. Returns (times, order)."""
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tg.LR, weight_decay=tg.WEIGHT_DECAY
    )
    fwd = torch.compile(model) if compile_model else model
    times, order = defaultdict(float), []

    @contextmanager
    def tic(name, record):
        if record and name not in order:
            order.append(name)
        _sync(device)
        t0 = time.perf_counter()
        yield
        _sync(device)
        if record:
            times[name] += time.perf_counter() - t0

    b, Lt = batch.shape[0], batch.shape[2]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for it in range(N_WARMUP + N_TIMED):
        record = it >= N_WARMUP
        optimizer.zero_grad()
        # Mirrors network_obar(): the input stage is the real config_inputs, so
        # it cannot drift out of sync with what training runs.
        with tic("config_inputs (W, T)", record):
            W, T = tg.config_inputs(batch, device)
        with tic("forward (GELT)", record):
            Obar = fwd(W, T).sum(dim=(1, 2, 3)).view(b, Lt)
        with tic("rayleigh_loss", record):
            loss, _, _ = tg.rayleigh_loss(Obar)
        with tic("backward", record):
            loss.backward()
        with tic("clip + optim.step", record):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    total = sum(times.values())
    print(f"\n── {label} ── per-step means over {N_TIMED} iters")
    for name in order:
        ms = times[name] / N_TIMED * 1e3
        print(f"  {name:<24} {ms:9.1f} ms   {100 * times[name] / total:5.1f}%")
    print(f"  {'TOTAL':<24} {total / N_TIMED * 1e3:9.1f} ms")
    steps_per_epoch = -(-int(tg.TRAIN_FRACTION * tg.N_CONFIGS) // b)
    s_per_epoch = total / N_TIMED * steps_per_epoch
    print(
        f"  ≈ {s_per_epoch:.0f} s/epoch at {steps_per_epoch} steps/epoch"
        f"  →  {s_per_epoch * tg.EPOCHS / 3600:.1f} h for {tg.EPOCHS} epochs"
        f" (the ens1 run early-stopped at 25)"
    )
    # The before-number, measured on the real thing rather than reconstructed:
    # logs/replication_ens1.log, 2026-07-04, 32 GB V100, BATCH_CONFIGS=6.
    print(
        "  reference: the ens1 run logged 7.77 s/step, 1942 s/epoch "
        "(V100, batch 6) — compare against that, not against a rebuilt baseline"
    )
    if device.type == "cuda":
        print(
            f"  peak CUDA memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"
        )
    return total / N_TIMED


def breakdown_inputs(batch, device):
    """Split config_inputs into its three stages.

    Timed separately rather than instrumented in place, so config_inputs stays
    the single definition of the input pipeline. The three should sum to roughly
    the ``config_inputs`` line above; a large gap means this reconstruction has
    drifted from it.
    """
    U = batch.to(device)
    b, Lt = batch.shape[0], batch.shape[2]
    levels = sorted(tg.INPUT_SMEAR_LEVELS)

    _sync(device)
    t0 = time.perf_counter()
    W, done = U, 0
    for lvl in levels:
        if lvl > done:
            W = ape_smear(W, tg.gaugegroup, alpha=tg.SMEAR_ALPHA, n_steps=lvl - done)
            done = lvl
    _sync(device)
    t_smear = time.perf_counter() - t0

    Usp = U[:, 1:].movedim(2, 1).contiguous()
    U3 = Usp.reshape(b * Lt, 3, tg.L, tg.L, tg.L, tg.NC, tg.NC)
    _sync(device)
    t0 = time.perf_counter()
    for _ in levels:
        plaquette_tensor(U3, tg.gaugegroup)
    _sync(device)
    t_plaq = time.perf_counter() - t0

    _sync(device)
    t0 = time.perf_counter()
    build_transport_average(U3, tg.R, tg.gaugegroup)
    _sync(device)
    t_T = time.perf_counter() - t0

    print("\n── config_inputs, stage by stage (one pass, same shapes) ──")
    tot = t_smear + t_plaq + t_T
    for name, t in (
        (f"ape_smear ladder {levels}", t_smear),
        (f"plaquette_tensor ×{len(levels)}", t_plaq),
        (f"build_transport_average R={tg.R}", t_T),
    ):
        print(f"  {name:<34} {t * 1e3:9.1f} ms   {100 * t / tot:5.1f}%")
    return t_smear


def legacy_smear_ladder(batch, device):
    """The APE ladder as it was built before 2026-09: a Python loop over configs
    with ``SU.project`` going through a batched 2×2 SVD + det.

    Kept here (and only here) so the change can be re-measured on any device
    rather than trusted. Bit-for-bit it agrees with the current route to
    complex64 rounding — 4.9e-7 max over the six-step ladder on real configs.
    """
    g = tg.gaugegroup

    def project_svd(M):
        Wm, _, Vh = torch.linalg.svd(M)
        Q = Wm @ Vh
        return Q / torch.linalg.det(Q).pow(1 / g.nc).unsqueeze(-1).unsqueeze(-1)

    def smear_once(U, alpha, n_steps):
        dirs = list(range(1, U.shape[1]))
        n_st = 2 * (len(dirs) - 1)
        out = U.clone()
        for _ in range(n_steps):
            new = out.clone()
            for cfg in range(out.shape[0]):
                for mu in dirs:
                    st = g.dagger(staple_sum(out[cfg], mu, g, nu_dirs=dirs))
                    new[cfg, mu] = project_svd(
                        (1 - alpha) * out[cfg, mu] + (alpha / n_st) * st
                    )
            out = new
        return out

    U = batch.to(device)
    _sync(device)
    t0 = time.perf_counter()
    W, done = U, 0
    for lvl in sorted(tg.INPUT_SMEAR_LEVELS):
        if lvl > done:
            W = smear_once(W, tg.SMEAR_ALPHA, lvl - done)
            done = lvl
    _sync(device)
    return time.perf_counter() - t0


def micro_bench(device, b):
    """Forward *and backward* of each attention stage, in isolation.

    The step timing says backward dominates but not which stage's backward. Two
    candidates behave very differently on a GPU and the difference decides
    whether the offset-loop rewrite is worth doing:

      * the neighbour gather ``KV[nb_indexer]``. Its backward is
        ``index_put_(accumulate=True)``, i.e. an atomic scatter-add with 25
        contributions landing on every destination element. The alternative is a
        ``roll`` per offset — 2× the forward traffic, but a plain coalesced copy
        in both directions and no atomics. Which wins is a hardware question.
      * the transport ``T · X · T†``. Two ``bmm`` calls with (nc, nc) @ (nc, H·d·nc)
        operands, i.e. millions of 2×2 matrices; its backward is four more. If
        this is the expensive half, the fix is the adjoint representation
        (T·X·T† is a real SO(3) rotation of X's traceless part for nc = 2), not
        the loop.

    Shapes are the production per-layer shapes at the given batch.
    """
    from gelt.blocks import GEMHSA

    Lt, L, NC = tg.LT, tg.L, tg.NC
    B = b * Lt
    blk = GEMHSA(
        tg.gaugegroup, L, 3, tg.R, d_input=tg.D_MODEL, nhead=tg.NHEAD,
        d_qkv=tg.D_QKV, dtype=tg.MODEL_DTYPE,
    ).to(device)
    n_off = blk.n_offsets
    KV = torch.randn(
        B, 2 * tg.NHEAD, tg.D_QKV, L, L, L, NC, NC,
        dtype=tg.MODEL_DTYPE, device=device, requires_grad=True,
    )
    T = torch.randn(B, n_off, L, L, L, NC, NC, dtype=tg.MODEL_DTYPE, device=device)
    T_dag = tg.gaugegroup.dagger(T)
    Q = torch.randn(
        B, tg.NHEAD, tg.D_QKV, L, L, L, NC, NC,
        dtype=tg.MODEL_DTYPE, device=device, requires_grad=True,
    )
    idx = tuple(blk._nbr_idx[k] for k in range(3))
    nb = (slice(None),) * 3 + idx + (slice(None), slice(None))

    def run(fn, n=3):
        for _ in range(2):  # warm-up
            out = fn()
            out.abs().pow(2).sum().backward()
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(n):
            out = fn()
        _sync(device)
        t_f = (time.perf_counter() - t0) / n
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(n):
            fn().abs().pow(2).sum().backward()
        _sync(device)
        return t_f, (time.perf_counter() - t0) / n - t_f

    lat = tuple(range(3, 6))
    stages = {
        "gather (advanced index)": lambda: KV[nb],
        "gather (roll + stack)": lambda: torch.stack(
            [
                torch.roll(KV, shifts=tuple(-c for c in off), dims=lat)
                for off in blk.offsets
            ],
            dim=3,
        ),
        "transport (2 bmm)": lambda: blk.transport(KV[nb].detach().requires_grad_(), T, T_dag),
        "rope_score": lambda: blk.rope_score(
            Q, blk.transport(KV[nb], T, T_dag)[:, : tg.NHEAD].detach().requires_grad_()
        ),
    }
    print("\n── per-stage forward / backward at the production per-layer shape ──")
    print(f"   B={B} slices, n_off={n_off}, channels={2 * tg.NHEAD * tg.D_QKV}, nc={NC}")
    for name, fn in stages.items():
        try:
            t_f, t_b = run(fn)
            print(
                f"  {name:<26} fwd {t_f * 1e3:8.1f} ms   bwd {t_b * 1e3:8.1f} ms"
                f"   (bwd/fwd {t_b / max(t_f, 1e-9):5.2f})"
            )
        except RuntimeError as exc:  # OOM at this shape is itself information
            print(f"  {name:<26} FAILED: {str(exc)[:70]}")


def main():
    torch.manual_seed(0)
    device = _device()
    print(f"device: {device}")
    b = int(sys.argv[1]) if len(sys.argv) > 1 else tg.BATCH_CONFIGS
    batch = _batch(b, device)
    print(
        f"batch {b} configs = {b * tg.LT} 3D slices of {tg.L}³, R={tg.R}, "
        f"layers={tg.GEMHSA_LAYERS}, d_model={tg.D_MODEL}, "
        f"smear levels {list(tg.INPUT_SMEAR_LEVELS)}, "
        f"grad_checkpoint={tg.GRAD_CHECKPOINT}"
    )

    model = _model(device)
    t_base = time_step(model, batch, device, "current")
    t_smear = breakdown_inputs(batch, device)

    if os.environ.get("PROFILE_DIAGNOSTICS") == "1":
        model.set_introspection(store_attention=True, diagnostics=True)
        t_diag = time_step(model, batch, device, "introspection stashes ON")
        model.set_introspection(store_attention=False, diagnostics=False)
        print(
            f"\n  introspection costs {(t_diag - t_base) * 1e3:.0f} ms/step "
            f"({100 * (t_diag / t_base - 1):.0f}%)"
        )

    if os.environ.get("PROFILE_LEGACY_SMEAR") == "1":
        t_old = legacy_smear_ladder(batch, device)
        print(
            f"\n── APE ladder, old route (config loop + SVD project) ──\n"
            f"  {t_old * 1e3:9.1f} ms  vs  {t_smear * 1e3:.1f} ms now  "
            f"→ {t_old / max(t_smear, 1e-9):.1f}× , "
            f"{(t_old - t_smear) * 1e3:.0f} ms/step saved "
            f"({100 * (t_old - t_smear) / t_base:.0f}% of the current step)"
        )

    if os.environ.get("PROFILE_MICRO", "1") == "1":
        micro_bench(device, b)

    if os.environ.get("PROFILE_COMPILE") == "1":
        try:
            t_c = time_step(_model(device), batch, device, "torch.compile",
                            compile_model=True)
            print(f"\n  torch.compile: {t_base / t_c:.2f}× vs eager")
        except Exception as exc:  # noqa: BLE001 — this is a probe, not a feature
            print(f"\n  torch.compile failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
