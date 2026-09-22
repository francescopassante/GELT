"""One (loop, size, seed) run of the 1+1D Wilson-loop regression.

The reproduction of Fig. 3 of PRL 128, 032003: supervised regression of the
traced Wilson loops ``W^(1x1)``, ``W^(1x2)``, ``W^(2x2)``, ``W^(4x4)`` on an
8 x 8 SU(2) lattice with the authors' own L-CNN. Their panels also carry a
baseline CNN; that half is deliberately not reproduced
(``notes/wilson_regression_1p1d.md`` §1).

Everything the Letter fixes is fixed here and nothing is tuned:

* **Architecture** from SM Table V, by name, with the parameter count checked
  against the table at build time.
* **Optimiser** AdamW with zero weight decay (SM SS VI.A). Learning rate, epoch
  cap and early-stopping patience come from ``TRAIN_HP`` --- 3e-3 / 20 / 5 for
  the two small loops, 1e-3 / 100 / 25 for the two large ones --- and the batch
  size is 50.
* **Loss** MSE on the per-site prediction against the per-site label. The L-CNN
  has no global average pool (SM SS V: "leaving out the lattice average for
  L-CNNs led to much easier training").
* **Selection** the best validation loss, which is also what early stopping
  watches, and the reported test numbers are that checkpoint's.

What is reported: ``mse_site`` (per lattice site) and ``mse_avg`` (both sides
averaged over the lattice first). **``mse_avg`` is the one to compare with the
Letter** --- Fig. 3 plots one point per configuration, and their own
``mse(global_average=True)`` reduces that way. ``mse_site`` is printed next to
it because a model can be right on average and wrong site by site, and the
averaged number alone cannot see the difference.

``WR_CONV_IMPL`` picks which L-CB parametrisation the stack is built from:
``exact`` (:mod:`gelt.lcnn_exact`, the parameter counts the Letter reports) or
``ref`` (the vendored code as published, one transported slot per layer wider).
See that module; neither is a correction of the other.

Artifacts: ``dumps/wilson_regression_<target>_<size>_<impl>_s<seed>.pt`` holds
the test-set predictions and labels (per site, so either MSE can be recomputed
offline), the loss history and the metadata.
``scripts/wilson_regression_figure.py`` assembles Fig. 3 from those.

Env overrides (also ``--name=value``): ``WR_TARGET`` (W11|W12|W22|W44),
``WR_SIZE`` (small|medium|large), ``WR_CONV_IMPL`` (exact|ref), ``WR_SEED``,
``WR_L``, ``WR_LR``, ``WR_EPOCHS``, ``WR_PATIENCE``, ``WR_BATCH``,
``WR_INIT_W``, ``WR_DEVICE``, ``WR_RUN_TAG``, ``WR_TEST_SIZES`` (evaluate the
trained model on larger lattices too).
"""

import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wilson_regression_common import (
    BATCH_SIZE,
    DUMP_DIR,
    GELT_ARCHS,
    GELT_LR,
    LCNN_NPARAM,
    PAPER_MSE_LCNN,
    TRAIN_HP,
    build_gelt,
    build_lcnn,
    build_transport,
    cfg,
    load_split,
    mse_pair,
    pick_device,
    real_dofs,
    validate_argv,
)


def make_batches(n, batch_size, shuffle):
    idx = torch.randperm(n) if shuffle else torch.arange(n)
    return [idx[i:i + batch_size] for i in range(0, n, batch_size)]


def run_split(model, data, batch_size, device, train_step=None):
    """One pass over a split. ``train_step`` makes it the training pass.

    ``data`` is ``(W, aux, y)``. Both architectures are called as
    ``model(W, aux)`` and differ only in what ``aux`` is: raw links for the
    reference L-CB, which transports internally, and the shortest-path-averaged
    L1-ball transport for GELT, which does not. Keeping the call signature
    identical is what makes the two arms the same experiment.

    Returns ``(mean loss, predictions)``; predictions are only accumulated when
    there is no optimiser step, since that is the only time they are wanted.
    """
    W, aux, y = data
    model.train() if train_step is not None else model.eval()
    tot, n, preds = 0.0, 0, []
    ctx = torch.enable_grad() if train_step is not None else torch.no_grad()
    with ctx:
        for b in make_batches(W.shape[0], batch_size, train_step is not None):
            out = model(W[b].to(device), aux[b].to(device))
            loss = nn.functional.mse_loss(out, y[b].to(device))
            if train_step is not None:
                train_step(loss)
            else:
                preds.append(out.cpu())
            tot += loss.item() * b.numel()
            n += b.numel()
    return tot / n, (torch.cat(preds) if preds else None)


def train(model, train_data, val_data, lr, epochs, patience, batch_size,
          device, ckpt_path):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    def step(loss):
        opt.zero_grad()
        loss.backward()
        opt.step()

    best, since_best = float("inf"), 0
    history = {"train": [], "val": []}
    t0 = time.time()
    for epoch in range(epochs):
        tr, _ = run_split(model, train_data, batch_size, device, train_step=step)
        vl, _ = run_split(model, val_data, batch_size, device)
        history["train"].append(tr)
        history["val"].append(vl)
        mark = ""
        if vl < best:
            best, since_best, mark = vl, 0, "  *"
            torch.save(model.state_dict(), ckpt_path)
        else:
            since_best += 1
        print(f"  epoch {epoch + 1:3d}/{epochs}  train {tr:.4e}  "
              f"val {vl:.4e}  ({time.time() - t0:.0f}s){mark}", flush=True)
        if since_best >= patience:
            print(f"  early stopping: {patience} epochs without improvement")
            break
    if best == float("inf"):
        raise RuntimeError(f"no checkpoint written to {ckpt_path}: val loss "
                           f"never finite. Check for NaN in the forward pass.")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return history, best


def main():
    validate_argv()
    arch = str(cfg("WR_ARCH", "lcnn")).lower()
    if arch not in ("lcnn", "gelt"):
        raise SystemExit(f"WR_ARCH must be 'lcnn' or 'gelt', got {arch!r}")
    target = str(cfg("WR_TARGET", "W11")).upper()
    size = str(cfg("WR_SIZE", "small" if arch == "lcnn" else "matched")).lower()
    conv_impl = str(cfg("WR_CONV_IMPL", "exact")).lower()
    seed = int(cfg("WR_SEED", 0))
    L = int(cfg("WR_L", 8))
    init_w = float(cfg("WR_INIT_W", 1.0))
    mlp_zero = str(cfg("WR_MLP_ZERO_INIT", "1")) not in ("0", "false", "False")
    tag = cfg("WR_RUN_TAG", "")
    extra_sizes = [int(s) for s in str(cfg("WR_TEST_SIZES", "")).split(",") if s]
    device = pick_device(cfg("WR_DEVICE", None))

    hp = TRAIN_HP[target]
    # The paper's rate is the L-CNN's. GELT's head is zero-initialised, so the
    # gradient reaches the attention only once the head has moved; GELT_LR is
    # what train_gelt.py measured for that stall, not a value tuned here.
    lr = float(cfg("WR_LR", hp["lr"] if arch == "lcnn" else GELT_LR))
    epochs = int(cfg("WR_EPOCHS", hp["epochs"]))
    patience = int(cfg("WR_PATIENCE", hp["patience"]))
    batch_size = int(cfg("WR_BATCH", BATCH_SIZE))

    os.makedirs(DUMP_DIR, exist_ok=True)
    torch.manual_seed(seed)

    R = GELT_ARCHS[(target, size)]["R"] if arch == "gelt" else None
    splits = {}
    for name in ("train", "val", "test"):
        U, W, y, beta, _meta = load_split(name, L=L, target=target)
        if arch == "gelt":
            print(f"  building transport R={R} for {name} "
                  f"({U.shape[0]} configurations)", flush=True)
            aux = build_transport(U, R=R, device=device, progress=False)
        else:
            aux = U
        splits[name] = (W, aux, y)
        if name == "test":
            test_beta = beta
    print(f"data L={L} target={target}: "
          + " ".join(f"{k}={v[0].shape[0]}" for k, v in splits.items()))

    if arch == "gelt":
        model, n_param, spec = build_gelt(target, size, L=L,
                                          mlp_zero_init=mlp_zero)
        paper_n = LCNN_NPARAM.get((target, "large"))
        impl = "-".join(f"{k}{v}" for k, v in spec.items())
        print(f"GELT {target} {size} seed={seed}: {n_param} real DOFs "
              f"(matched against Table V's L-CNN: {paper_n})")
        print(f"  {spec}  mlp_zero_init={mlp_zero}")
    else:
        model, n_param = build_lcnn(target, size, L=L, conv_impl=conv_impl,
                                    init_w=init_w)
        paper_n = LCNN_NPARAM.get((target, size))
        impl = conv_impl
        print(f"L-CNN {target} {size} [{conv_impl}] seed={seed}: "
              f"{n_param} parameters (Table V: {paper_n})")
    model = model.to(device)
    print(f"  AdamW lr={lr} wd=0  epochs<={epochs} patience={patience} "
          f"batch={batch_size}  device={device}")

    stem = (f"wilson_regression_{target}_{size}_{conv_impl}_s{seed}"
            if arch == "lcnn"
            else f"wilson_regression_gelt_{target}_{size}_s{seed}")
    if tag:
        stem += f"_{tag}"
    ckpt = os.path.join(DUMP_DIR, stem + ".pth")
    history, best_val = train(model, splits["train"], splits["val"], lr, epochs,
                              patience, batch_size, device, ckpt)

    _, pred = run_split(model, splits["test"], batch_size, device)
    mse_site, mse_avg = mse_pair(pred, splits["test"][2])
    ref = PAPER_MSE_LCNN[target]
    # The paper's number is the L-CNN's. It is printed for the GELT arm too, as
    # the scale of the problem, and is *not* a reproduction target there.
    label = "paper" if arch == "lcnn" else "paper's L-CNN"
    print(f"\n  test MSE (lattice-averaged, Fig. 3's convention): {mse_avg:.3e}"
          f"   [{label}: {ref:.1e}]")
    print(f"  test MSE (per site):                             {mse_site:.3e}")
    print(f"  best val loss: {best_val:.4e}")

    dump = {
        "arch": arch, "target": target, "size": size, "seed": seed,
        "conv_impl": conv_impl if arch == "lcnn" else None,
        "gelt_spec": GELT_ARCHS[(target, size)] if arch == "gelt" else None,
        "L": L, "n_param": n_param, "paper_n_param": paper_n,
        "lr": lr, "epochs": epochs, "patience": patience, "batch": batch_size,
        "init_w": init_w, "mlp_zero_init": mlp_zero, "run_tag": tag or None,
        "history": history, "best_val": best_val,
        "mse_site": mse_site, "mse_avg": mse_avg, "paper_mse": ref,
        "test_pred": pred, "test_true": splits["test"][2], "test_beta": test_beta,
        "epochs_run": len(history["train"]),
    }

    # The same trained model on larger lattices: translational equivariance plus
    # a per-site head means an L-CNN transfers without retraining, which is one
    # of the Letter's claims and costs nothing to check here.
    if extra_sizes and arch == "gelt":
        # GEMHSA bakes the lattice extents into its offset index maps at
        # construction; there is no update_dims for it, so the volume-transfer
        # check is an L-CNN-only reading here rather than a silent no-op.
        print(f"  [transfer] WR_TEST_SIZES is ignored for WR_ARCH=gelt: "
              f"GEMHSA's offset maps are built for L={L}")
        extra_sizes = []
    for Lx in extra_sizes:
        try:
            Ux, Wx, yx, bx, _ = load_split("test", L=Lx, target=target)
        except SystemExit as e:
            print(f"  [L={Lx}] skipped: {e}")
            continue
        model.update_dims(Lx)
        _, px = run_split(model, (Ux, Wx, yx), batch_size, device)
        ss, sa = mse_pair(px, yx)
        print(f"  [L={Lx}] test MSE avg {sa:.3e}  site {ss:.3e}")
        dump[f"transfer_L{Lx}"] = {"mse_site": ss, "mse_avg": sa,
                                   "pred": px, "true": yx, "beta": bx}
    if extra_sizes:
        model.update_dims(L)

    path = os.path.join(DUMP_DIR, stem + ".pt")
    torch.save(dump, path)
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
