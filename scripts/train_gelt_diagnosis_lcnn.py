from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from gelt.blocks_routed import GELT
from gelt.lattice import rectangular_wilson_loop
from gelt.sampler import mcmc_ensemble


def averaged_wilson_loop(config, gaugegroup, R, T, mu, nu):
    """Lattice-averaged Re Tr W(R, T) / nc — one scalar per config."""
    y = rectangular_wilson_loop(config, gaugegroup, R=R, T=T, mu=mu, nu=nu)
    return y.mean(dim=tuple(range(1, y.ndim)))


"""
========================================================================================
 L-CNN-replication training script for the routed GELT.

 This is the routed three-branch architecture (gelt/blocks_routed.py) configured to
 collapse onto Favoni et al.'s L-CNN: pattern="A,C,A,C,A,C,A,C" (alternating L-Conv
 and L-Bilin layers, NO attention branch instantiated), residual_skip + l_act left ON
 for optimization stability (the strict L-CNN replication is
 ``residual_skip=False, l_act=False``, exposed below as a one-line toggle).

 Goal: verify that the new architecture, when reduced to pure L-Conv + L-Bilin, can
 reach at least L-CNN's known capability on SU(2) — Favoni et al. report 4×4 Wilson
 loops are learnable. If this run cannot reproduce that result, the regression isn't
 architectural and the issue is elsewhere (signal/statistics, optimization, target
 scale). If it CAN, the parallel three-branch design has a residual issue that the
 sequential L-Conv → L-Bilin pattern fixes.

 Differences from train_gelt_diagnosis_routed.py:
   * pattern="A,C,A,C,A,C,A,C" — sequential L-Conv → L-Bilin, NOT parallel A+C+B.
   * Branch B (attention) is never instantiated, so it cannot accidentally engage.
   * Per-layer α diagnostic only prints α_A on 'A' layers, α_C on 'C' layers.
========================================================================================
"""


def evaluate(model, test_loader, criterion, device, save_outputs=False, progress=True):
    model.eval()

    test_loss = 0.0
    test_count = 0
    if save_outputs:
        all_targets = []
        all_outputs = []
    iterator = tqdm(test_loader) if progress else test_loader
    with torch.no_grad():
        for X, T, y in iterator:
            X, T, y = X.to(device), T.to(device), y.to(device)
            outputs = model(X, T)
            loss = criterion(outputs, y)
            batch_size = y.shape[0]
            test_loss += loss.item() * batch_size
            test_count += batch_size
            if save_outputs:
                all_targets.append(y.cpu())
                all_outputs.append(outputs.cpu())

    test_loss /= test_count
    if save_outputs:
        all_targets = torch.cat(all_targets)
        all_outputs = torch.cat(all_outputs)
        return test_loss, all_targets, all_outputs
    return test_loss


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epochs,
    patience=5,
    checkpoint_path: str = "best_model.pth",
):
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    epochs_no_improve = 0

    epoch_bar = tqdm(range(epochs))
    for epoch in epoch_bar:
        model.train()
        train_loss = 0.0
        train_count = 0
        # Dead-signal diagnostics (same as the other diagnostic trainers):
        # output collapse → constant predictor; grad-norm dropouts → no
        # learning signal reaching the parameters.
        out_sum = 0.0
        out_sq_sum = 0.0
        out_count = 0
        grad_norm_sum = 0.0
        grad_norm_batches = 0
        first_batch = True
        for X, T, y in train_loader:
            X, T, y = X.to(device), T.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X, T)
            loss = criterion(outputs, y)
            loss.backward()

            with torch.no_grad():
                o = outputs.detach()
                out_sum += o.sum().item()
                out_sq_sum += (o * o).sum().item()
                out_count += o.numel()
                total_sq = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        total_sq += p.grad.detach().pow(2).sum().item()
                grad_norm_sum += total_sq**0.5
                grad_norm_batches += 1
                if first_batch and epoch < 3:
                    epoch_bar.write(
                        f"  ep {epoch:>3d} [first batch]  "
                        f"y μ={y.mean().item():+.4f} σ={y.std(unbiased=False).item():.4f}  "
                        f"out μ={o.mean().item():+.4f} σ={o.std(unbiased=False).item():.4f}  "
                        f"loss={loss.item():.4f}"
                    )
                    first_batch = False

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_size = y.shape[0]
            train_loss += loss.item() * batch_size
            train_count += batch_size

        train_loss /= train_count
        train_losses.append(train_loss)
        scheduler.step()

        out_mean = out_sum / out_count
        out_var = max(out_sq_sum / out_count - out_mean * out_mean, 0.0)
        out_std = out_var**0.5
        avg_grad_norm = grad_norm_sum / max(grad_norm_batches, 1)

        val_loss = evaluate(model, val_loader, criterion, device, progress=False)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        epoch_bar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}")

        # Per-layer α: each block has only the α's for its active branches.
        # 'A' blocks expose α_A; 'C' blocks expose α_C; 'B'-containing blocks
        # expose α_B. Outer α is present only when residual_skip=True.
        if hasattr(model, "blocks"):
            entries = []
            for layer in model.blocks:
                tags = [f"[{layer.branches}]"]
                if hasattr(layer, "alpha"):
                    tags.append(f"α={layer.alpha.item():+.3f}")
                if hasattr(layer, "alpha_A"):
                    tags.append(f"A={layer.alpha_A.item():+.3f}")
                if hasattr(layer, "alpha_B"):
                    tags.append(f"B={layer.alpha_B.item():+.3f}")
                if hasattr(layer, "alpha_C"):
                    tags.append(f"C={layer.alpha_C.item():+.3f}")
                entries.append(" ".join(tags))
            fc2_std = model.mlp.fc2.weight.detach().abs().mean().item()
            epoch_bar.write(
                f"  ep {epoch + 1:>3d}  train={train_loss:.4f}  val={val_loss:.4f}  "
                f"out μ={out_mean:+.4f} σ={out_std:.4f}  "
                f"‖grad‖={avg_grad_norm:.2e}  |fc2|̄={fc2_std:.4f}\n"
                f"    layers: {' | '.join(entries)}"
            )

        if epochs_no_improve >= patience:
            epoch_bar.write(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    if best_val_loss == float("inf"):
        raise RuntimeError(
            f"No checkpoint written to {checkpoint_path}: val_loss never improved "
            f"over inf. Likely NaN/Inf losses — check the training output."
        )

    return train_losses, val_losses, epoch + 1


if __name__ == "__main__":
    torch.manual_seed(0)
    from gelt import SU, Z2, build_plaquette_datasets

    D = 2
    L = 8
    gaugegroup = SU(2)
    # L-CNN replication uses transports anchored at corners; R=2 with a
    # depth-8 alternating stack gives Manhattan reach 16 — comfortably
    # covers a 4×4 Wilson loop (corner-to-corner Manhattan distance 8)
    # and matches Favoni et al.'s reported reach on similar targets.
    R = 2
    model_dtype = torch.float32 if isinstance(gaugegroup, Z2) else torch.complex64

    beta = 2.0
    # Start at the 4×4 Wilson loop — the target Favoni et al. report L-CNN
    # solving. If this run can't reproduce that, the issue isn't architectural.
    # Bump down to 3×3 / 2×3 if you want to confirm the easier targets first.
    loop_R, loop_T, mu, nu = 4, 4, 0, 1
    dataset_parameters = {
        "N": 2000,
        "D": D,
        "L": L,
        "gaugegroup": gaugegroup,
        "R": R,
        "splits": [0.7, 0.15, 0.15],
        "save": False,
        "prefix": (
            f"su2_plaquette_L{L}_D{D}_N2000_beta{beta}_R{R}"
            f"_wloop{loop_R}x{loop_T}_lcnn"
        ),
        "structured": True,
        "sampler": mcmc_ensemble,
        "beta": beta,
        # Lattice-averaged target — the weaker / easier supervision (codex §1).
        # Use reduction="mean" on the model. Flip ``target`` to
        # ``rectangular_wilson_loop`` and reduction="none" for per-site.
        "target": partial(averaged_wilson_loop, R=loop_R, T=loop_T, mu=mu, nu=nu),
        "n_therm": 200,
        "n_skip": 5,
        "dtype": torch.complex64,
        # Single canonical shortest path. Non-abelian path averaging is
        # non-unitary (codex §3) — strict L-CNN needs clean group-valued
        # transports.
        "transport_mode": "single",
    }

    train_parameters = {
        "lr": 3e-3,
        "batch_size": 64,
        "epochs": 3000,
        "patience": 3000,
        "checkpoint_path": "best_gelt_lcnn.pth",
    }

    # L-CNN-replication GELT:
    #   pattern="A,C,A,C,A,C,A,C" — 4 L-Conv + 4 L-Bilin layers, alternating.
    #     Sequential ordering: each L-Bilin sees the L-Conv output (transported
    #     features), unlike the parallel A+B+C in the default routed block.
    #   residual_skip=True, l_act=True (defaults) — keep residual + L-Act gate
    #     for optimization stability. The codex notes (§6) argue that residuals
    #     break the multiplicative-coupling chicken-and-egg at random init;
    #     they don't change expressivity, only optimizability.
    #     Set residual_skip=False, l_act=False for STRICT L-CNN (output =
    #     α·branch(W), no residual add, no gate).
    #   constructive_A=True — Branch A initialised on a single-offset basis
    #     of routes, so the L-Conv backbone is active from epoch 0.
    model_parameters = {
        "gaugegroup": gaugegroup,
        "L": L,
        "D": D,
        "R": R,
        "nhead": 2,
        "pattern": "A,C,A,C,A,C,A,C",
        "d_qkv": 16,
        "gate": "softplus",
        "dtype": model_dtype,
        "mlp_hidden": 64,
        "mlp_out": 1,
        # Lattice-averaged supervision pairs with "mean" reduction.
        "reduction": "mean",
        # Outer ReZero ON; warm-start so the block engages from epoch 0
        # (same logic as train_gelt_diagnosis.py).
        "alpha_init": 1.0,
        # Per-branch warm-starts: only A and C matter under this pattern.
        # α_A=1 (constructive basis active immediately), α_C=1 (L-Bilin
        # multiplies the L-Conv output from depth 2 onward).
        "alpha_A_init": 1.0,
        "alpha_B_init": 0.0,  # unused — pattern has no 'B' layers
        "alpha_C_init": 1.0,
        "init_scale": 1.0,    # no need to lift attention scores — no attention.
        "mlp_zero_init": False,
        # Wider residual stream so L-Bilin has channel diversity to multiply.
        # The plaquette input is D(D-1)/2 = 1 channel in D=2; channel mixing
        # cannot conjure diversity from a single input channel, so widening
        # via ChannelLift is essential. 32 ≈ Favoni et al.'s L-CNN width.
        "d_model": 32,
        "constructive_A": True,
        "trilinear": False,
        # ----- strict L-CNN switch ------------------------------------------
        # Flip both to False to drop the residual ReZero and the L-Act gate.
        # Strict L-CNN replication; harder to optimize but mathematically
        # the closest match to Favoni et al.'s architecture.
        "residual_skip": True,
        "l_act": True,
    }

    train_dataset, val_dataset, test_dataset = build_plaquette_datasets(
        **dataset_parameters
    )

    y_train = train_dataset.dataset.tensors[-1][train_dataset.indices]
    mu_y = y_train.mean()
    sigma_y = y_train.std(unbiased=False).clamp_min(1e-12)
    train_dataset.dataset.tensors[-1].sub_(mu_y).div_(sigma_y)
    print(f"target scaler fit: μ_y = {mu_y.item():.4f} | σ_y = {sigma_y.item():.4f}")

    model = GELT(**model_parameters)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=train_parameters["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_parameters["batch_size"],
        shuffle=True,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=train_parameters["batch_size"], shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=train_parameters["batch_size"], shuffle=False
    )

    X, T, y = next(iter(train_loader))
    n_params = sum(p.numel() for p in model.parameters())
    n_real_dofs = sum(
        p.numel() * (2 if p.is_complex() else 1) for p in model.parameters()
    )
    print(
        f"GELT (L-CNN replication) | pattern={model.pattern} | "
        f"residual_skip={model_parameters['residual_skip']} l_act={model_parameters['l_act']} | "
        f"params: {n_params:,} ({n_real_dofs:,} real DOFs) | "
        f"X {tuple(X.shape)} {X.dtype} | T {tuple(T.shape)} {T.dtype} | "
        f"out {tuple(model(X, T).shape)}"
    )

    model = model.to(device)

    train_losses, val_losses, full_epochs = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=train_parameters["epochs"],
        patience=train_parameters["patience"],
        checkpoint_path=train_parameters["checkpoint_path"],
    )

    model.load_state_dict(
        torch.load(
            train_parameters["checkpoint_path"], map_location=device, weights_only=True
        )
    )

    test_loss, all_targets, all_outputs = evaluate(
        model, test_loader, criterion, device, save_outputs=True
    )

    all_targets = all_targets * sigma_y + mu_y
    all_outputs = all_outputs * sigma_y + mu_y
    test_label_var = all_targets.var(unbiased=False).item()
    test_mse_physical = ((all_outputs - all_targets) ** 2).mean().item()
    test_r2 = (
        1.0 - test_mse_physical / test_label_var if test_label_var > 0 else float("nan")
    )

    print(
        f"Test Loss (norm): {test_loss:.4f} | "
        f"Test MSE (physical): {test_mse_physical:.4f} | "
        f"Var(y): {test_label_var:.4f} | R²: {test_r2:.4f}"
    )

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.yscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss (L-CNN replication of routed GELT)")
    plt.legend()
    plt.grid(True)
    plt.savefig("gelt_lcnn_loss.png", dpi=150, bbox_inches="tight")
    plt.close()

    t_flat = all_targets.reshape(-1).numpy()
    o_flat = all_outputs.reshape(-1).numpy()
    if t_flat.size > 20000:
        rng = torch.Generator().manual_seed(0)
        idx = torch.randperm(t_flat.size, generator=rng)[:20000].numpy()
        t_flat, o_flat = t_flat[idx], o_flat[idx]
    plt.figure(figsize=(8, 8))
    plt.scatter(t_flat, o_flat, alpha=0.5, s=4)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title("True vs Predicted Values (L-CNN replication of routed GELT, Test Set)")
    plt.grid(True)
    plt.savefig("gelt_lcnn_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()

    results = {
        "test_loss": test_loss,
        "test_label_var": test_label_var,
        "test_r2": test_r2,
        "epochs": full_epochs,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "pattern": model.pattern,
        "residual_skip": model_parameters["residual_skip"],
        "l_act": model_parameters["l_act"],
    }
