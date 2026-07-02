"""Timing breakdown of one train_glueball.py training step.

Answers "where does the step time go?" — in particular whether the on-the-fly
``build_transport_average`` (audit item 4: no precomputed-T dataset) is the
bottleneck, or the GEMHSA forward/backward. Reuses train_glueball.py's
constants, model construction, and loss so the numbers reflect the real step.

Uses the cached ensemble if present, otherwise Haar-random configs (identical
compute cost — the timings don't depend on the link values).

Run (from the repo root):
    python scripts/profile_glueball_step.py [batch_configs]

``batch_configs`` defaults to train_glueball.BATCH_CONFIGS (8); pass e.g. 16
to probe a bigger batch's step time/memory, or 2 on a small local GPU.
"""

import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_glueball as tg

from gelt.blocks_rope import GELT
from gelt.lattice import build_transport_average, plaquette_tensor, random_links

N_WARMUP = 2  # untimed iterations (JIT/cudnn autotune, allocator warm-up)
N_TIMED = 5


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def main():
    torch.manual_seed(0)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"device: {device}")

    b = int(sys.argv[1]) if len(sys.argv) > 1 else tg.BATCH_CONFIGS
    if os.path.exists(tg.CACHE):
        print(f"using first {b} configs of cached ensemble {tg.CACHE}")
        batch = torch.load(tg.CACHE)[:b].to(tg.MODEL_DTYPE)
    else:
        print(f"cache absent — using {b} Haar-random configs (same compute cost)")
        batch = torch.stack(
            [random_links(tg.L, tg.D, tg.gaugegroup, Lt=tg.LT) for _ in range(b)]
        ).to(tg.MODEL_DTYPE)

    model = GELT(
        gaugegroup=tg.gaugegroup, L=tg.L, D=3, R=tg.R, nhead=tg.NHEAD,
        gemhsa_layers=tg.GEMHSA_LAYERS, d_qkv=tg.D_QKV, gate=tg.GATE,
        dtype=tg.MODEL_DTYPE, mlp_hidden=tg.MLP_HIDDEN, mlp_out=1,
        reduction="none", init_scale=tg.INIT_SCALE, qk_init_scale=tg.QK_INIT_SCALE,
        mlp_zero_init=False, d_model=tg.D_MODEL,
        grad_checkpoint=tg.GRAD_CHECKPOINT,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tg.LR, weight_decay=tg.WEIGHT_DECAY)

    times = defaultdict(float)
    order = []  # section names in execution order

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

    Lt, L, NC = tg.LT, tg.L, tg.NC
    for it in range(N_WARMUP + N_TIMED):
        record = it >= N_WARMUP
        optimizer.zero_grad()
        # Mirrors network_obar(), split into its stages (+ H2D move).
        with tic("slice→3D + H2D", record):
            Usp = batch[:, 1:].movedim(2, 1).contiguous()
            U3 = Usp.reshape(b * Lt, 3, L, L, L, NC, NC).to(device)
        with tic("plaquette_tensor", record):
            W = plaquette_tensor(U3, tg.gaugegroup)
        with tic("build_transport", record):
            T = build_transport_average(U3, tg.R, tg.gaugegroup)
        with tic("forward (GELT)", record):
            Obar = model(W, T).sum(dim=(1, 2, 3)).view(b, Lt)
        with tic("rayleigh_loss", record):
            loss, _, _ = tg.rayleigh_loss(Obar)
        with tic("backward", record):
            loss.backward()
        with tic("clip + optim.step", record):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    total = sum(times.values())
    print(
        f"\nper-step means over {N_TIMED} iters "
        f"(batch {b} configs = {b * Lt} 3D slices, R={tg.R}, "
        f"layers={tg.GEMHSA_LAYERS}):"
    )
    for name in order:
        ms = times[name] / N_TIMED * 1e3
        print(f"  {name:<20} {ms:9.1f} ms   {100 * times[name] / total:5.1f}%")
    print(f"  {'TOTAL':<20} {total / N_TIMED * 1e3:9.1f} ms")
    steps_per_epoch = -(-tg.TRAIN_FRACTION * tg.N_CONFIGS // b)  # ≈, for orientation
    print(
        f"  ≈ {total / N_TIMED * steps_per_epoch:.0f} s/epoch at "
        f"~{steps_per_epoch:.0f} steps/epoch"
    )
    if device.type == "cuda":
        print(
            f"peak CUDA memory: "
            f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB "
            f"(the BATCH_CONFIGS knob — try 16 if this is well under 16 GiB)"
        )


if __name__ == "__main__":
    main()
