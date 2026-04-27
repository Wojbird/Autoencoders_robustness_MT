import csv
import os

import torch
from torch.utils.data import DataLoader

from evaluation.evaluate import evaluate_reconstruction
from utils.helpers import EvalResult


def _unwrap_tensor(x):
    if isinstance(x, (tuple, list)):
        return x[0]
    return x


def _write_csv_header(path: str, fieldnames: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def _append_csv_row(path: str, fieldnames: list[str], row: dict) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


def _forward_noisy_latent(
    model,
    x: torch.Tensor,
    noise_std: float,
) -> torch.Tensor:
    if hasattr(model, "encode") and callable(getattr(model, "encode")) and \
       hasattr(model, "decode") and callable(getattr(model, "decode")):
        z = model.encode(x)
        z = _unwrap_tensor(z)
        z_noisy = z + noise_std * torch.randn_like(z)
        x_hat = model.decode(z_noisy)
        return _unwrap_tensor(x_hat)

    if hasattr(model, "encoder") and hasattr(model, "decoder"):
        z = model.encoder(x)
        z = _unwrap_tensor(z)
        z_noisy = z + noise_std * torch.randn_like(z)
        x_hat = model.decoder(z_noisy)
        return _unwrap_tensor(x_hat)

    raise AttributeError(
        f"{model.__class__.__name__} must expose either encode()/decode() "
        f"or encoder/decoder for noisy_latent training."
    )


def _assert_classic_autoencoder_only(model) -> None:
    if hasattr(model, "vq_loss") or hasattr(model, "vq_losses"):
        raise ValueError(
            "train_noisy_latent_model() supports only classic autoencoders. "
            "VQ-VAE/VQ-VAE2 models are intentionally not supported."
        )


def train_noisy_latent_model(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: torch.device,
    results_dir: str,
    checkpoints_dir: str,
    model_name: str,
):
    _assert_classic_autoencoder_only(model)

    model = model.to(device)

    epochs = int(config.get("epochs", 50))
    learning_rate = float(config.get("learning_rate", 1e-4))
    weight_decay = float(config.get("weight_decay", 0.0))
    noise_std = float(config.get("noise_std", 0.1))
    eval_input_noise_std = float(config.get("eval_noise_std", noise_std))
    patience = int(config.get("early_stopping_patience", 10))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    loss_fn = torch.nn.MSELoss()

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    csv_path = os.path.join(results_dir, f"{model_name}_noisy_latent_metrics.csv")
    fieldnames = [
        "epoch",
        "train_loss",
        "val_clean_loss",
        "val_clean_mse",
        "val_clean_psnr",
        "val_clean_ssim",
        "val_input_noisy_loss",
        "val_input_noisy_mse",
        "val_input_noisy_psnr",
        "val_input_noisy_ssim",
        "val_latent_noisy_loss",
        "val_latent_noisy_mse",
        "val_latent_noisy_psnr",
        "val_latent_noisy_ssim",
    ]
    _write_csv_header(csv_path, fieldnames)

    best_val_latent_noisy_loss = float("inf")
    epochs_without_improvement = 0

    best_ckpt_path = os.path.join(checkpoints_dir, f"{model_name}_noisy_latent_best.pt")
    last_ckpt_path = os.path.join(checkpoints_dir, f"{model_name}_noisy_latent_last.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n_batches = 0

        for x, _ in train_loader:
            x = x.to(device)

            optimizer.zero_grad(set_to_none=True)

            x_hat = _forward_noisy_latent(
                model,
                x,
                noise_std=noise_std,
            )
            x_hat = torch.clamp(x_hat, 0.0, 1.0)

            loss = loss_fn(x_hat, x)
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.item())
            n_batches += 1

        if n_batches == 0:
            raise ValueError("Empty train_loader in train_noisy_latent_model().")

        train_loss = train_loss_sum / n_batches

        val_clean: EvalResult = evaluate_reconstruction(
            model,
            val_loader,
            device,
            variant="clean",
            noise_std=0.0,
            latent_noise=False,
            noise_seed=1234,
        )

        val_input_noisy: EvalResult = evaluate_reconstruction(
            model,
            val_loader,
            device,
            variant="noisy",
            noise_std=eval_input_noise_std,
            latent_noise=False,
            noise_seed=1234,
        )

        val_latent_noisy: EvalResult = evaluate_reconstruction(
            model,
            val_loader,
            device,
            variant="noisy_latent",
            noise_std=noise_std,
            latent_noise=True,
            noise_seed=1234,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_clean_loss": val_clean.loss,
            "val_clean_mse": val_clean.mse,
            "val_clean_psnr": val_clean.psnr,
            "val_clean_ssim": val_clean.ssim,
            "val_input_noisy_loss": val_input_noisy.loss,
            "val_input_noisy_mse": val_input_noisy.mse,
            "val_input_noisy_psnr": val_input_noisy.psnr,
            "val_input_noisy_ssim": val_input_noisy.ssim,
            "val_latent_noisy_loss": val_latent_noisy.loss,
            "val_latent_noisy_mse": val_latent_noisy.mse,
            "val_latent_noisy_psnr": val_latent_noisy.psnr,
            "val_latent_noisy_ssim": val_latent_noisy.ssim,
        }
        _append_csv_row(csv_path, fieldnames, row)

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "model_name": model_name,
                "input_variant": "noisy_latent",
                "selection_metric": "last",
                "metrics": row,
            },
            last_ckpt_path,
        )

        if val_latent_noisy.loss < best_val_latent_noisy_loss:
            best_val_latent_noisy_loss = val_latent_noisy.loss
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "model_name": model_name,
                    "input_variant": "noisy_latent",
                    "selection_metric": "val_latent_noisy_loss",
                    "metrics": row,
                },
                best_ckpt_path,
            )
        else:
            epochs_without_improvement += 1

        print(
            f"[{model_name} | noisy_latent] "
            f"Epoch {epoch:03d}/{epochs} | "
            f"train_loss={train_loss:.6f} | "
            f"val_clean_loss={val_clean.loss:.6f} | "
            f"val_input_noisy_loss={val_input_noisy.loss:.6f} | "
            f"val_latent_noisy_loss={val_latent_noisy.loss:.6f} | "
            f"val_latent_noisy_psnr={val_latent_noisy.psnr:.3f} | "
            f"val_latent_noisy_ssim={val_latent_noisy.ssim:.4f}"
        )

        if epochs_without_improvement >= patience:
            print(
                f"[{model_name} | noisy_latent] Early stopping after {epoch} epochs. "
                f"Best val_latent_noisy_loss={best_val_latent_noisy_loss:.6f}."
            )
            break

    return best_ckpt_path