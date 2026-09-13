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
    PROFILE_ROPE=1         also run the rope_score bench: every candidate on K̃
                           as transport returns it (a permuted view) and on a
                           contiguous copy, against a measured bandwidth. Decides
                           between notes/performance_audit.md §5.2 (layout),
                           §5.3 (the contraction) and §5.6 (torch.compile).
                           PROFILE_ROPE_COMPILE=0 drops the compiled candidates.
"""

import math
import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_glueball as tg

from gelt.glueball import ape_smear
from gelt.lattice import build_transport_average, plaquette_tensor, random_links
from gelt.lcnn import build_axis_transports
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
    # tg._build_model() is the training loop's own constructor, so this cannot
    # drift from what the run does — and it profiles whichever architecture
    # GLUEBALL_ARCH selects (the L-CNN baseline included).
    return tg._build_model().to(device)


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
        with tic(f"forward ({tg.NET})", record):
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
    if tg.ARCH == "lcnn":
        build_axis_transports(U3, tg.LCNN_K, tg.gaugegroup)
        t_label = f"build_axis_transports K={tg.LCNN_K}"
    else:
        build_transport_average(U3, tg.R, tg.gaugegroup)
        t_label = f"build_transport_average R={tg.R}"
    _sync(device)
    t_T = time.perf_counter() - t0

    print("\n── config_inputs, stage by stage (one pass, same shapes) ──")
    tot = t_smear + t_plaq + t_T
    for name, t in (
        (f"ape_smear ladder {levels}", t_smear),
        (f"plaquette_tensor ×{len(levels)}", t_plaq),
        (t_label, t_T),
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

    if device.type == "cuda":
        torch.cuda.empty_cache()
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

    # Each stage is built by a *setup* that runs before the clock starts and
    # whose inputs die with the closure, so no stage is timed with another
    # stage's work inside it and none of them pin 5 GiB for the whole table.
    #
    # The 2026-09-13 run measured `rope_score` at 238 ms this way and the
    # backlog was re-ranked around it (notes/performance_audit.md §5.0(iv)).
    # It was wrong: the lambda read
    #     blk.rope_score(Q, blk.transport(KV[nb], ...)[...].detach().requires_grad_())
    # and `.detach().requires_grad_()` severs the *backward* graph, not the
    # forward work — so the number was gather + transport + rope_score, and only
    # the reported `bwd` belonged to rope_score alone. Setup goes in the setup.
    def setup_gather():
        return lambda: KV[nb]

    def setup_roll():
        return lambda: torch.stack(
            [
                torch.roll(KV, shifts=tuple(-c for c in off), dims=lat)
                for off in blk.offsets
            ],
            dim=3,
        )

    def setup_transport(heads=2 * tg.NHEAD):
        X = KV[:, :heads][nb].detach().requires_grad_()
        return lambda: blk.transport(X, T, T_dag)

    def setup_rope():
        K = blk.transport(KV[nb], T, T_dag)[:, : tg.NHEAD].detach().requires_grad_()
        return lambda: blk.rope_score(Q, K)

    stages = [
        ("gather (advanced index)", setup_gather, None),
        ("gather (roll + stack)", setup_roll, None),
        # transport's three intermediates are each the size of its input, so the
        # full KV call needs ~19 GiB and OOMs on a 32 GiB card next to the step's
        # reservation. It is bandwidth-bound and linear in the channel count, so
        # the fallback runs the K half and the printout says to double it.
        ("transport (2 bmm)", setup_transport, lambda: setup_transport(tg.NHEAD)),
        ("rope_score", setup_rope, None),
    ]
    print("\n── per-stage forward / backward at the production per-layer shape ──")
    print(f"   B={B} slices, n_off={n_off}, channels={2 * tg.NHEAD * tg.D_QKV}, nc={NC}")
    for name, setup, fallback in stages:
        # Release the caching allocator's free blocks between stages. The step
        # above reserved ~20 GiB; its activations are gone but the reserved
        # arena is fragmented, and `transport` wants a single 4.45 GiB block —
        # which is how it OOM'd here while running fine inside the step.
        if device.type == "cuda":
            torch.cuda.empty_cache()
        note = ""
        fn = None
        try:
            fn = setup()
            t_f, t_b = run(fn)
        except RuntimeError as exc:  # OOM at this shape is itself information
            fn = None
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if fallback is None:
                print(f"  {name:<26} FAILED: {str(exc)[:70]}")
                continue
            try:
                fn = fallback()
                t_f, t_b = run(fn)
                note = "   [half the channels — double for the production KV call]"
            except RuntimeError as exc2:
                print(f"  {name:<26} FAILED: {str(exc2)[:70]}")
                continue
        finally:
            del fn
        print(
            f"  {name:<26} fwd {t_f * 1e3:8.1f} ms   bwd {t_b * 1e3:8.1f} ms"
            f"   (bwd/fwd {t_b / max(t_f, 1e-9):5.2f}){note}"
        )


def _rope_score_two_bracket(blk, Q, K_tilde):
    """The retired two-bracket route, kept as the A/B reference.

    This is what ``GEMHSA.rope_score`` was before 2026-09-13: the Frobenius
    product taken against Q and against Q's pair-swapped copy in two separate
    multiply-reduce pairs, streaming K̃ — 2.389 GB at the production shape —
    once per bracket. The single-pass version measured 1.30× against it here and
    has landed, so this stays only so the win can be re-measured (and so a
    regression shows up as the gap closing). Pinned against the live
    ``rope_score`` in ``_check_rope_candidates``.
    """
    B, H, dq = Q.shape[0], Q.shape[1], Q.shape[2]
    spatial = Q.shape[3:-2]
    nc = Q.shape[-1]
    n = K_tilde.shape[3]
    P = blk.n_pairs

    Qp = Q.reshape(B, H, P, 2, *spatial, nc, nc)
    Q_swap = torch.stack((Qp[:, :, :, 1], -Qp[:, :, :, 0]), dim=3).reshape(
        B, H, dq, *spatial, nc, nc
    )

    def brackets(Qx):
        t = (Qx.unsqueeze(3).conj() * K_tilde).sum(dim=(-2, -1))
        return t.reshape(B, H, P, 2, n, *spatial).sum(dim=3)

    freq = blk.rope_freq.real if blk.rope_freq.is_complex() else blk.rope_freq
    disp = blk._rope_disp.real if blk._rope_disp.is_complex() else blk._rope_disp
    angle = disp * freq
    view = (1, 1, P, n) + (1,) * len(spatial)
    cos = torch.cos(angle).t().reshape(view)
    sin = torch.sin(angle).t().reshape(view)
    score = (cos * brackets(Q).real + sin * brackets(Q_swap).real).sum(dim=2)
    return score / math.sqrt(dq * nc)


def _rope_score_real(blk, Q, K_tilde):
    """The single pass again, in real arithmetic — the fusable candidate.

    ``Re Σ conj(Q)·K = Q.real·K.real + Q.imag·K.imag``, so the score never needs
    a complex multiply: the current route does four real multiplies per element
    and throws half the result away in ``.real``. Two consequences.

    *Eager*: the product tensor is float32 where it was complex64, so the
    trailing variant axis doubles a tensor that halved — the intermediate is
    |K̃| bytes, the same transient the two-bracket route already pays, for
    ~3.1·|K̃| of traffic against its ~7.2.

    *Compiled*: Inductor warns that it "does not support code generation for
    complex operators" (§5.6's first obstacle, reproduced by this bench), so a
    compiled complex candidate cannot fuse the multiply into the reduction. This
    one is real float32 end to end — a plain multiply-reduce, which is exactly
    what Inductor does fuse. If any candidate reaches the ~1× floor it is this
    one compiled.
    """
    B, H, dq = Q.shape[0], Q.shape[1], Q.shape[2]
    spatial = Q.shape[3:-2]
    nc = Q.shape[-1]
    n = K_tilde.shape[3]
    P = blk.n_pairs

    Qp = Q.reshape(B, H, P, 2, *spatial, nc, nc)
    Q_swap = torch.stack((Qp[:, :, :, 1], -Qp[:, :, :, 0]), dim=3)
    Q2 = torch.stack((Qp, Q_swap), dim=-1).unsqueeze(4)
    Kp = K_tilde.reshape(B, H, P, 2, n, *spatial, nc, nc).unsqueeze(-1)
    # No .conj(): Re(conj(Q)·K) = Qr·Kr + Qi·Ki identically.
    t = (Q2.real * Kp.real + Q2.imag * Kp.imag).sum(dim=(3, -3, -2))

    freq = blk.rope_freq.real if blk.rope_freq.is_complex() else blk.rope_freq
    disp = blk._rope_disp.real if blk._rope_disp.is_complex() else blk._rope_disp
    angle = disp * freq
    view = (1, 1, P, n) + (1,) * len(spatial)
    cos = torch.cos(angle).t().reshape(view)
    sin = torch.sin(angle).t().reshape(view)
    return (cos * t[..., 0] + sin * t[..., 1]).sum(dim=2) / math.sqrt(dq * nc)


def _check_rope_candidates(device):
    """Pin every candidate against ``GEMHSA.rope_score`` at a toy shape.

    Cheap (L=4, B=2) and run before the timings: a bench of a wrong kernel is
    worse than no bench. ``rope_score`` itself is pinned against the naive
    ``apply_rope``-then-Frobenius oracle in tests/test_blocks.py.
    """
    from gelt.blocks import GEMHSA

    L, nc, H, dq, B = 4, tg.NC, 2, 6, 2
    blk = GEMHSA(
        tg.gaugegroup, L, 3, 1, d_input=4, nhead=H, d_qkv=dq, dtype=tg.MODEL_DTYPE
    ).to(device)
    n = blk.n_offsets
    Q = torch.randn(B, H, dq, L, L, L, nc, nc, dtype=tg.MODEL_DTYPE, device=device)
    X = torch.randn(B, 2 * H, dq, n, L, L, L, nc, nc, dtype=tg.MODEL_DTYPE, device=device)
    T = torch.randn(B, n, L, L, L, nc, nc, dtype=tg.MODEL_DTYPE, device=device)
    # K̃ exactly as attend() sees it: a permuted, non-contiguous view.
    K = blk.transport(X, T, tg.gaugegroup.dagger(T))[:, :H]
    assert not K.is_contiguous(), "transport stopped returning a view — re-read this bench"
    ref = blk.rope_score(Q, K)
    for name, got in (
        ("two bracket (retired)", _rope_score_two_bracket(blk, Q, K)),
        ("single pass, real", _rope_score_real(blk, Q, K)),
        ("single pass, real, contiguous K̃", _rope_score_real(blk, Q, K.contiguous())),
        ("current, contiguous K̃", blk.rope_score(Q, K.contiguous())),
        ("two bracket, contiguous K̃", _rope_score_two_bracket(blk, Q, K.contiguous())),
    ):
        err = (got - ref).abs().max().item() / max(ref.abs().max().item(), 1e-12)
        assert err < 1e-5, f"{name} disagrees with rope_score: rel {err:.2e}"
    return True


def rope_bench(device, b):
    """Why is ``rope_score`` 238 ms? Three hypotheses, one run.

    The 2026-09-13 profile put ``rope_score`` at 237.9 ms per layer forward —
    over four layers, essentially the whole 1154 ms forward.
    notes/performance_audit.md §5.0 reads that as a *traffic* problem (K̃ is
    streamed once per Frobenius bracket) and §5.3 proposes the single-pass
    rewrite. The roofline says traffic cannot be the whole story: K̃ is 2.389 GB
    at the production per-layer shape, the two-bracket route moves ~7.2× that =
    17.2 GB, which is 19 ms at the V100's 900 GB/s. Measured 238 ms is **12×
    off**.

    The third hypothesis, which §5 does not have: ``GEMHSA.transport`` returns
    ``out.permute(inv_perm)``, a **non-contiguous view**. Its logical innermost
    lattice axis carries stride nc·H·d_qkv·nc = 48 elements while the colour row
    index carries 24 and the colour column 1, so every downstream elementwise
    kernel reads K̃ (and Ṽ) uncoalesced — a warp walking the contiguous output
    touches ~8 cache lines where it should touch 2. If that is the 12×, the next
    item is §5.2 (the Q/K/V memory layout), not §5.3, and the single-pass
    rewrite buys far less than its byte count promises.

    So: every candidate is timed on K̃ **as transport returns it** and on a
    contiguous copy, against a measured device bandwidth and a pure-reduction
    floor. The same A/B runs on the value path ``(α·Ṽ).sum(n)``, which reads the
    other half of the same permuted tensor.

    Forward and backward separately — the step is 64% backward.
    """
    from gelt.blocks import GEMHSA

    if device.type == "cuda":
        torch.cuda.empty_cache()
    _check_rope_candidates(device)

    Lt, L, NC = tg.LT, tg.L, tg.NC
    B = b * Lt
    blk = GEMHSA(
        tg.gaugegroup, L, 3, tg.R, d_input=tg.D_MODEL, nhead=tg.NHEAD,
        d_qkv=tg.D_QKV, dtype=tg.MODEL_DTYPE,
    ).to(device)
    n_off, H, dq = blk.n_offsets, tg.NHEAD, tg.D_QKV

    def fresh(requires_grad=True):
        """A K̃ with transport's layout, without paying for transport."""
        base = torch.randn(
            B, n_off, L, L, L, NC, H, dq, NC,
            dtype=tg.MODEL_DTYPE, device=device, requires_grad=requires_grad,
        )
        D = 3
        inv_perm = (0, 3 + D, 4 + D, 1) + tuple(range(2, 2 + D)) + (2 + D, 5 + D)
        return base, base.permute(*inv_perm)

    base, K_strided = fresh()
    Q = torch.randn(
        B, H, dq, L, L, L, NC, NC,
        dtype=tg.MODEL_DTYPE, device=device, requires_grad=True,
    )
    alpha = torch.randn(B, H, n_off, L, L, L, device=device)
    nbytes = K_strided.numel() * K_strided.element_size()

    # Measured, not assumed: a large contiguous copy is read+write, so the
    # achieved bandwidth is 2·nbytes/t. Everything below is quoted against it.
    probe = torch.empty_like(base)
    for _ in range(2):
        probe.copy_(base.detach())
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(3):
        probe.copy_(base.detach())
    _sync(device)
    bw = 3 * 2 * nbytes / (time.perf_counter() - t0)
    del probe

    print("\n── rope_score: traffic, layout, or fusion? ──")
    print(
        f"   K̃ = {nbytes / 1e9:.3f} GB per layer "
        f"(B={B}, H={H}, d_qkv={dq}, n_off={n_off}, {L}³, nc={NC})"
    )
    print(f"   measured copy bandwidth {bw / 1e9:.0f} GB/s → K̃ read once = "
          f"{nbytes / bw * 1e3:.1f} ms, the 7.2× two-bracket route = "
          f"{7.2 * nbytes / bw * 1e3:.1f} ms")

    def run(fn, n=3):
        for _ in range(2):
            fn().abs().pow(2).sum().backward()
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        _sync(device)
        t_f = (time.perf_counter() - t0) / n
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(n):
            fn().abs().pow(2).sum().backward()
        _sync(device)
        return t_f, (time.perf_counter() - t0) / n - t_f

    K_contig = K_strided.detach().contiguous().requires_grad_()
    alpha_b = alpha.unsqueeze(2).unsqueeze(-1).unsqueeze(-1)

    compiled = None
    if os.environ.get("PROFILE_ROPE_COMPILE", "1") == "1":
        try:
            compiled = torch.compile(GEMHSA.rope_score, dynamic=False)
            compiled_real = torch.compile(_rope_score_real, dynamic=False)
        except Exception as exc:  # noqa: BLE001 — a probe, not a dependency
            print(f"   torch.compile unavailable: {str(exc)[:60]}")

    stages = [
        ("floor: K̃.sum(colour)  strided", lambda: K_strided.sum(dim=(-2, -1))),
        ("floor: K̃.sum(colour)  contig ", lambda: K_contig.sum(dim=(-2, -1))),
        ("K̃.contiguous() (the copy)    ", lambda: K_strided.contiguous()),
        ("rope_score   current  strided", lambda: blk.rope_score(Q, K_strided)),
        ("rope_score   current  contig ", lambda: blk.rope_score(Q, K_contig)),
        ("rope_score   2-brckt  strided", lambda: _rope_score_two_bracket(blk, Q, K_strided)),
        ("rope_score   2-brckt  contig ", lambda: _rope_score_two_bracket(blk, Q, K_contig)),
        ("rope_score   1-pass-re strided", lambda: _rope_score_real(blk, Q, K_strided)),
        ("rope_score   1-pass-re contig ", lambda: _rope_score_real(blk, Q, K_contig)),
        ("value  (α·Ṽ).sum(n)   strided", lambda: (alpha_b * K_strided).sum(dim=3)),
        ("value  (α·Ṽ).sum(n)   contig ", lambda: (alpha_b * K_contig).sum(dim=3)),
    ]
    if compiled is not None:
        stages += [
            ("rope_score   cmp-cplx strided", lambda: compiled(blk, Q, K_strided)),
            ("rope_score   cmp-cplx contig ", lambda: compiled(blk, Q, K_contig)),
            ("rope_score   cmp-real strided", lambda: compiled_real(blk, Q, K_strided)),
            ("rope_score   cmp-real contig ", lambda: compiled_real(blk, Q, K_contig)),
        ]

    for name, fn in stages:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        try:
            t_f, t_b = run(fn)
        except RuntimeError as exc:  # OOM at this shape is itself information
            print(f"  {name:<30}  FAILED: {str(exc)[:60]}")
            continue
        # "×roof" = how many times K̃ this stage's forward would have to read to
        # justify its time at the measured bandwidth. A pure reduction should
        # land near 1; the two-bracket route's traffic budget is 7.2.
        roof = t_f * bw / nbytes
        print(
            f"  {name:<30}  fwd {t_f * 1e3:7.1f} ms   bwd {t_b * 1e3:7.1f} ms"
            f"   (bwd/fwd {t_b / max(t_f, 1e-9):4.2f}, {roof:5.1f}× K̃ read)"
        )


def main():
    torch.manual_seed(0)
    device = _device()
    print(f"device: {device}")
    # Positional argv is the batch size; `--name=value` flags belong to
    # train_glueball's own env/argv overrides (e.g. --arch=lcnn), which it
    # parsed at import — skip them here rather than choking on them.
    positional = [a for a in sys.argv[1:] if not a.startswith("--")]
    b = int(positional[0]) if positional else tg.BATCH_CONFIGS
    batch = _batch(b, device)
    print(
        f"batch {b} configs = {b * tg.LT} 3D slices of {tg.L}³, "
        f"{tg.NET} ({tg._geometry_label()}), "
        f"smear levels {list(tg.INPUT_SMEAR_LEVELS)}, "
        f"grad_checkpoint={tg.GRAD_CHECKPOINT}"
    )

    model = _model(device)
    t_base = time_step(model, batch, device, "current")
    t_smear = breakdown_inputs(batch, device)

    if os.environ.get("PROFILE_DIAGNOSTICS") == "1" and tg.ARCH == "gelt":
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

    # The micro-bench times GEMHSA's stages; there is nothing to A/B in an
    # L-CNN step beyond the whole-step and input-stage numbers above.
    if os.environ.get("PROFILE_MICRO", "1") == "1" and tg.ARCH == "gelt":
        micro_bench(device, b)

    if os.environ.get("PROFILE_ROPE") == "1" and tg.ARCH == "gelt":
        rope_bench(device, b)

    if os.environ.get("PROFILE_COMPILE") == "1":
        try:
            t_c = time_step(_model(device), batch, device, "torch.compile",
                            compile_model=True)
            print(f"\n  torch.compile: {t_base / t_c:.2f}× vs eager")
        except Exception as exc:  # noqa: BLE001 — this is a probe, not a feature
            print(f"\n  torch.compile failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
