import os
import wandb
import torch
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__name__)

from data.data_setter import get_subnet_datasets, get_imagenet_datasets
from evaluation.evaluate import evaluate_reconstruction, compute_training_loss
from utils.helpers import (
    get_device,
    save_images,
    plot_metrics,
    load_config,
    ensure_val_fraction,
    init_csv_logger,
    EarlyStopping,
    safe_save_state_dict,
)


def _unwrap_tensor(x):
    if isinstance(x, (tuple, list)):
        return x[0]
    return x


def _forward_with_latent_noise(model, x, noise_std: float):
    """
    Forward pass for classical autoencoders only:
    x -> encode -> z -> z_noisy -> decode -> x_hat
    """
    if hasattr(model, "encode") and callable(getattr(model, "encode")) and \
       hasattr(model, "decode") and callable(getattr(model, "decode")):
        z = model.encode(x)
        z = _unwrap_tensor(z)
        z_noisy = z + torch.randn_like(z) * noise_std
        x_hat = model.decode(z_noisy)
        x_hat = _unwrap_tensor(x_hat)
        return x_hat

    if hasattr(model, "vq_loss") or hasattr(model, "vq_losses") or hasattr(model, "get_vq_losses"):
        z = model.encoder(x)
        z = _unwrap_tensor(z)
        z_noisy = z + torch.randn_like(z) * noise_std
        x_hat = model.decoder(z_noisy)
        x_hat = _unwrap_tensor(x_hat)
        return x_hat

    raise AttributeError(
        f"{model.__class__.__name__} must expose either "
        f"encode()/decode() or encoder/decoder for noisy_latent training."
    )


def train_noisy_latent_model(
    model_class,
    config_path: str,
    *,
    dataset_type: str = "subset",
    log: bool = False,
    gpu_id: int | None = None,
    log_wandb: bool = False,
) -> str:
    """
    Trains with noise added in latent space:
    x -> z -> z_noisy -> x_hat

    Supports only classical autoencoders.
    Does NOT support VQ-VAE / VQ-VAE2.
    """
    cfg = load_config(config_path)

    model_name = cfg["name"]
    image_size = int(cfg["image_size"])
    batch_size = int(cfg["batch_size"])
    num_workers = int(cfg.get("num_workers", 4))
    epochs = int(cfg["epochs"])
    lr = float(cfg["learning_rate"])
    wd = float(cfg.get("weight_decay", 0.0))
    noise_std = float(cfg.get("noise_latent", cfg.get("noise_std", 0.0)))
    val_fraction = float(cfg.get("val_subset_fraction", 1.0))
    patience = int(cfg.get("early_stopping_patience", 10))
    scheduler_factor = float(cfg.get("scheduler_factor", 0.5))
    scheduler_patience = int(cfg.get("scheduler_patience", 5))
    scheduler_min_lr = float(cfg.get("scheduler_min_lr", 1e-6))
    scheduler_threshold = float(cfg.get("scheduler_threshold", 1e-4))

    device = get_device(gpu_id)
    try:
        model = model_class(cfg).to(device)
    except TypeError as e:
        raise TypeError(
            f"{model_class.__name__} must accept config dict in __init__(self, config). "
            f"Original error: {e}"
        )

    if hasattr(model, "vq_loss") or hasattr(model, "vq_losses"):
        raise TypeError(
            f"{model_class.__name__} appears to be a VQ-based model. "
            f"train_noisy_latent_model supports only classical autoencoders."
        )

    if dataset_type == "full":
        train_set, val_set = get_imagenet_datasets(image_size=image_size)
    elif dataset_type == "subset":
        train_set, val_set = get_subnet_datasets(image_size=image_size, val_split=0.1)
        val_set = ensure_val_fraction(val_set, val_fraction)
    else:
        raise ValueError("dataset_type must be 'subset' or 'full'.")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience,
        threshold=scheduler_threshold,
        min_lr=scheduler_min_lr,
    )

    results_dir = os.path.join("results", model_name, dataset_type, "noisy_latent")
    os.makedirs(results_dir, exist_ok=True)

    ckpt_dir = os.path.join("checkpoints", model_name, dataset_type, "noisy_latent")
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_path = os.path.join(ckpt_dir, f"{model_name}_noisy_latent_best.pt")
    csv_path = os.path.join(results_dir, "metrics_per_epoch.csv")
    csv_f, csv_writer = init_csv_logger(csv_path)

    metrics_hist = {
        "mse_train": [], "psnr_train": [], "ssim_train": [],
        "mse_val": [], "psnr_val": [], "ssim_val": [],
        "loss_train": [], "loss_val": [],
    }

    es = EarlyStopping(patience=patience, min_delta=0.0)
    best_val = float("inf")
    best_epoch = 0

    run = None
    if log_wandb:
        run = wandb.init(
            project="autoencoders-robustness",
            name=f"{model_name}_{dataset_type}_noisy_latent",
            config={
                **cfg,
                "dataset_type": dataset_type,
                "training_variant": "noisy_latent",
                "model_class": model_class.__name__,
                "gpu_id": gpu_id,
                "noise_std": noise_std,
            },
        )

    try:
        for epoch in range(1, epochs + 1):
            # ---- Train
            model.train()
            train_loss_sum = 0.0
            n_batches = 0

            for x, _ in train_loader:
                x = x.to(device)

                optimizer.zero_grad(set_to_none=True)
                x_hat = _forward_with_latent_noise(model, x, noise_std)
                x_hat = torch.clamp(x_hat, 0.0, 1.0)

                loss = compute_training_loss(model, x_hat, x, allow_vq=False)
                loss.backward()
                optimizer.step()

                train_loss_sum += float(loss.item())
                n_batches += 1

            train_loss = train_loss_sum / max(1, n_batches)

            # ---- Train metrics
            train_eval = evaluate_reconstruction(
                model,
                train_loader,
                device,
                variant="noisy_latent",
                noise_std=noise_std,
                latent_noise=True,
                max_batches=20,
            )

            # ---- Validation
            val_eval = evaluate_reconstruction(
                model,
                val_loader,
                device,
                variant="noisy_latent",
                noise_std=noise_std,
                latent_noise=True,
            )

            scheduler.step(val_eval.loss)
            current_lr = optimizer.param_groups[0]["lr"]

            metrics_hist["loss_train"].append(train_loss)
            metrics_hist["loss_val"].append(val_eval.loss)

            metrics_hist["mse_train"].append(train_eval.mse)
            metrics_hist["psnr_train"].append(train_eval.psnr)
            metrics_hist["ssim_train"].append(train_eval.ssim)

            metrics_hist["mse_val"].append(val_eval.mse)
            metrics_hist["psnr_val"].append(val_eval.psnr)
            metrics_hist["ssim_val"].append(val_eval.ssim)

            csv_writer.writerow([
                epoch,
                train_loss, train_eval.mse, train_eval.psnr, train_eval.ssim,
                val_eval.loss, val_eval.mse, val_eval.psnr, val_eval.ssim,
            ])
            csv_f.flush()

            is_best = val_eval.loss < best_val
            if is_best:
                best_val = val_eval.loss
                best_epoch = epoch
                state_dict_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                safe_save_state_dict(state_dict_cpu, ckpt_path)

            if log:
                logger.info(
                    f"Epoch {epoch}/{epochs} | "
                    f"train_loss={train_loss:.6f} | "
                    f"val_loss={val_eval.loss:.6f} | "
                    f"PSNR={val_eval.psnr:.2f} | "
                    f"SSIM={val_eval.ssim:.4f} | "
                    f"lr={current_lr:.6e} | "
                    f"noise_std={noise_std:.4f}"
                )

            if log_wandb:
                wandb.log({
                    "epoch": epoch,
                    "lr": current_lr,
                    "noise_std": noise_std,

                    "train/loss": train_loss,
                    "train/mse": train_eval.mse,
                    "train/psnr": train_eval.psnr,
                    "train/ssim": train_eval.ssim,

                    "val/loss": val_eval.loss,
                    "val/mse": val_eval.mse,
                    "val/psnr": val_eval.psnr,
                    "val/ssim": val_eval.ssim,

                    "best/val_loss": best_val,
                    "best/epoch": best_epoch,
                })

            plot_metrics(metrics_hist, results_dir)

            images_dir = os.path.join(results_dir, "images")
            os.makedirs(images_dir, exist_ok=True)

            image_path = os.path.join(images_dir, f"recon_epoch_{epoch:04d}.png")
            save_images(
                model=model,
                dataloader=val_loader,
                device=device,
                save_path=image_path,
                num_images=8,
                add_noise=False,
                latent_noise=True,
                noise_std=noise_std,
            )

            if log_wandb:
                wandb.log({
                    "val/reconstructions": wandb.Image(
                        image_path,
                        caption=f"{model_name} | noisy_latent | epoch {epoch}"
                    )
                })

                if is_best:
                    wandb.run.summary["best_val_loss"] = val_eval.loss
                    wandb.run.summary["best_val_mse"] = val_eval.mse
                    wandb.run.summary["best_val_psnr"] = val_eval.psnr
                    wandb.run.summary["best_val_ssim"] = val_eval.ssim
                    wandb.run.summary["best_epoch"] = epoch
                    wandb.run.summary["best_checkpoint_path"] = ckpt_path
                    wandb.run.summary["noise_std"] = noise_std

            if es.step(val_eval.loss):
                if log:
                    logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

        txt_path = os.path.join(results_dir, "metrics_per_epoch.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            for i in range(len(metrics_hist["loss_val"])):
                f.write(
                    f"epoch={i+1} "
                    f"train_loss={metrics_hist['loss_train'][i]:.6f} "
                    f"val_loss={metrics_hist['loss_val'][i]:.6f} "
                    f"val_mse={metrics_hist['mse_val'][i]:.6f} "
                    f"val_psnr={metrics_hist['psnr_val'][i]:.6f} "
                    f"val_ssim={metrics_hist['ssim_val'][i]:.6f}\n"
                )

    finally:
        csv_f.close()
        if log_wandb:
            wandb.finish()

    return ckpt_path