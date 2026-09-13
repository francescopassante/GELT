"""Our L-CNN block against the reference implementation's, same shape, same box.

`lge-cnn-master/` is the authors' code for Favoni et al. (2012.12901). Its pure
-PyTorch layer (`lge_cnn/nn/layers.py`, `LConvBilin`) is the honest opponent for
`gelt/lcnn.py`: the repo also ships hand-written numba-CUDA kernels
(`layers_cuda.py`, forward *and* backward) for exactly these operations, and no
PyTorch implementation should be expected to match those.

What this times, per block, forward and forward+backward separately, at the
glueball task's production shape (B = configs x timeslices slices of 12³):

  ours   `gelt.lcnn.LCB`            — L-Conv + L-Bilin, our parametrisation
  ref    `LConvBilin`               — the merged layer, their parametrisation

The two are NOT the same function: theirs is a single 3-index bilinear kernel
weight[n_out, n_in1, n_in2] over [1, W, W†] and its transported copies, ours is
that kernel factored through L-Conv's c_out (a low-rank version of the same
family). Only the cost is comparable — read this for ms and GiB, never as a
numerical cross-check.

Their layer is loaded straight from the file: `lge_cnn/nn/__init__.py` pulls in
h5py, which this repo does not depend on, and `layers.py` itself needs only
torch and numpy.

Run:
    python scripts/bench_lcnn_reference.py              # production shape
    python scripts/bench_lcnn_reference.py 24 6 2       # B, channels, K
"""

import importlib.util
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gelt.lattice import SU, random_links
from gelt.lcnn import LCB, build_axis_transports

REF_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "lge-cnn-master", "lge_cnn", "nn", "layers.py",
)


def _load_reference():
    if not os.path.exists(REF_PATH):
        return None
    spec = importlib.util.spec_from_file_location("_lge_layers", REF_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time(fn, device, n=3):
    """(forward ms, forward+backward ms), best of n after one warm-up."""
    fwd, both = [], []
    for i in range(n + 1):
        _sync(device)
        t0 = time.perf_counter()
        out = fn()
        _sync(device)
        t1 = time.perf_counter()
        loss = out.real.pow(2).sum() if out.is_complex() else out.pow(2).sum()
        loss.backward()
        _sync(device)
        t2 = time.perf_counter()
        if i:  # skip the warm-up
            fwd.append(t1 - t0)
            both.append(t2 - t0)
    return 1e3 * min(fwd), 1e3 * min(both)


def main():
    device = _device()
    args = [int(a) for a in sys.argv[1:] if not a.startswith("--")]
    B = args[0] if args else 144          # 6 configs x 24 timeslices
    C = args[1] if len(args) > 1 else 5   # the shootout's c_hidden
    K = args[2] if len(args) > 2 else 2   # hops per axis, both orientations
    L, D, nc = 12, 3, 2
    dtype = torch.complex64
    print(f"device {device} | B={B} slices of {L}³, channels={C}, K={K}, nc={nc}")

    gg = SU(nc)
    torch.manual_seed(0)
    U = torch.stack(
        [random_links(L=L, D=D, gaugegroup=gg, dtype=dtype) for _ in range(B)]
    ).to(device)
    W = torch.randn(B, C, L, L, L, nc, nc, dtype=dtype, device=device)

    results = []

    # ── ours ────────────────────────────────────────────────────────────────
    T = build_axis_transports(U, K, gg)
    block = LCB(gg, c_in=C, c_out=C, D=D, K=K, dtype=dtype).to(device)
    W_ours = W.clone().requires_grad_(True)
    try:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        f, b = _time(lambda: block(W_ours, T), device)
        peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else float("nan")
        results.append(("ours  gelt.lcnn.LCB", f, b, peak))
    except torch.cuda.OutOfMemoryError:
        results.append(("ours  gelt.lcnn.LCB", float("nan"), float("nan"), float("nan")))
        torch.cuda.empty_cache()

    # ── theirs ──────────────────────────────────────────────────────────────
    ref = _load_reference()
    if ref is None:
        print(f"reference implementation not found at {REF_PATH} — skipped")
    else:
        # Their layout: x = cat(u, w) on the channel axis, sites flattened, and
        # a trailing real/imaginary axis (view_as_complex).
        X = L**D
        u_ref = torch.view_as_real(
            U.movedim(1, -3).reshape(B, X, D, nc, nc)
        ).contiguous().to(device)
        w_ref = torch.view_as_real(
            W.movedim(1, -3).reshape(B, X, C, nc, nc)
        ).contiguous().to(device).requires_grad_(True)
        # kernel_size = K + 1: their symmetric kernel_range is [-(ks-1), ks-1],
        # so ks = K + 1 is K hops each way — our K. dilation must be >= 1 (their
        # LConvBilin's transport loop is `for step in range(dilation)`).
        layer = ref.LConvBilin(
            dims=[L] * D, kernel_size=K + 1, dilation=1, n_in=C, n_out=C, nc=nc
        ).to(device)
        try:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            f, b = _time(
                lambda: layer(torch.cat((u_ref, w_ref), dim=2))[:, :, D:], device
            )
            peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else float("nan")
            results.append(("ref   LConvBilin (v2)", f, b, peak))
        except torch.cuda.OutOfMemoryError:
            results.append(("ref   LConvBilin (v2)", float("nan"), float("nan"), float("nan")))
            torch.cuda.empty_cache()

    print(f"\n{'block':<24}{'forward':>12}{'fwd+bwd':>12}{'peak GiB':>12}")
    for name, f, b, peak in results:
        print(f"{name:<24}{f:>10.1f} ms{b:>10.1f} ms{peak:>12.2f}")
    if len(results) == 2 and results[0][2] == results[0][2]:  # not NaN
        print(
            f"\nfwd+bwd ratio ours/ref: {results[0][2] / results[1][2]:.2f}× "
            f"(a 4-layer step is ~4 of these, run three times under checkpointing)"
        )


if __name__ == "__main__":
    main()
