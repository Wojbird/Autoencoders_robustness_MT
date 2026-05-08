import os
import wandb
import torch
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__name__)

from data.data_setter import get_subnet_datasets, get_imagenet_datasets
from evaluation.evaluate import evaluate_reconstruction, compute_training_loss
from training._adversarial import (
    unwrap_tensor,
    build_discriminator,
    discriminator_step,
    generator_adv_loss,
)
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


def train_clean_model(
    model_class,
    config_path: str,
    *,
    dataset_type: str = "subset",
    log: bool = False,
    gpu_id: int | None = None,
    log_wandb: bool = False,
) -> str:
    cfg = load_config(config_path)

    model_name = cfg["name"]
    image_size = int(cfg["image_size"])
    batch_size = int(cfg["batch_size"])
    num_workers = int(cfg.get("num_workers", 4))
    epochs = int(cfg["epochs"])
    lr = float(cfg["learning_rate"])
    wd = float(cfg.get("weight_decay", 0.0))
    eval_noise_std = float(cfg.get("eval_noise_std", cfg.get("noise_std", 0.0)))
    val_fraction = float(cfg.get("val_subset_fraction", 1.0))
    train_eval_fraction = float(cfg.get("train_eval_fraction", 0.1))
    patience = int(cfg.get("early_stopping_patience", 10))
    scheduler_factor = float(cfg.get("scheduler_factor", 0.5))
    scheduler_patience = int(cfg.get("scheduler_patience", 5))
    scheduler_min_lr = float(cfg.get("scheduler_min_lr", 1e-6))
    scheduler_threshold = float(cfg.get("scheduler_threshold", 1e-4))

    device = get_device(gpu_id)
    model = model_class(cfg).to(device)

    discriminator, optimizer_d, bce, adv_weight = build_discriminator(model, cfg, device)

    if dataset_type == "full":
        train_set, val_set = get_imagenet_datasets(image_size=image_size)
    elif dataset_type == "subset":
        train_set, val_set = get_subnet_datasets(image_size=image_size, val_split=0.1)
        val_set = ensure_val_fraction(val_set, val_fraction)
    else:
        raise ValueError("dataset_type must be 'subset' or 'full'.")

    train_eval_set = ensure_val_fraction(train_set, train_eval_fraction, split_seed=123)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    train_eval_loader = DataLoader(
        train_eval_set,
        batch_size=batch_size,
        shuffle=False,
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

    results_dir = os.path.join("results", model_name, dataset_type, "clean")
    ckpt_dir = os.path.join("checkpoints", model_name, dataset_type, "clean")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_path = os.path.join(ckpt_dir, f"{model_name}_clean_best.pt")
    csv_path = os.path.join(results_dir, "metrics_per_epoch.csv")

    csv_f, csv_writer = init_csv_logger(
        csv_path,
        extra_columns=[
            "val_input_noisy_loss",
            "val_input_noisy_mse",
            "val_input_noisy_psnr",
            "val_input_noisy_ssim",
            "disc_loss",
        ],
        overwrite=True,
    )

    metrics_hist = {
        "mse_train": [], "psnr_train": [], "ssim_train": [],
        "mse_val": [], "psnr_val": [], "ssim_val": [],
        "loss_train": [], "loss_val": [],
    }

    es = EarlyStopping(patience=patience, min_delta=0.0)
    best_val_clean = float("inf")
    best_epoch = 0

    run = None
    if log_wandb:
        run = wandb.init(
            settings=wandb.Settings(init_timeout=180, start_method="thread"),
            project="autoencoders-robustness",
            name=f"{model_name}_{dataset_type}_clean",
            config={
                **cfg,
                "dataset_type": dataset_type,
                "training_variant": "clean",
                "model_class": model_class.__name__,
                "gpu_id": gpu_id,
                "selection_metric": "val_clean_loss",
            },
        )

    try:
        for epoch in range(1, epochs + 1):
            model.train()
            if discriminator is not None:
                discriminator.train()

            train_loss_sum = 0.0
            disc_loss_sum = 0.0
            n_batches = 0

            for x, _ in train_loader:
                x = x.to(device)

                if discriminator is not None:
                    with torch.no_grad():
                        x_hat_disc = unwrap_tensor(model(x))
                    disc_loss_sum += discriminator_step(
                        discriminator,
                        optimizer_d,
                        bce,
                        x,
                        x_hat_disc,
                    )

                optimizer.zero_grad(set_to_none=True)

                x_hat = unwrap_tensor(model(x))
                x_hat = torch.clamp(x_hat, 0.0, 1.0)

                loss = compute_training_loss(
                    model,
                    x_hat,
                    x,
                    allow_vq=True,
                )

                if discriminator is not None:
                    loss = loss + adv_weight * generator_adv_loss(discriminator, bce, x_hat)

                loss.backward()
                optimizer.step()

                train_loss_sum += float(loss.item())
                n_batches += 1

            train_loss = train_loss_sum / max(1, n_batches)
            disc_loss = disc_loss_sum / max(1, n_batches) if discriminator is not None else 0.0

            train_eval = evaluate_reconstruction(
                model,
                train_eval_loader,
                device,
                variant="clean",
                noise_std=0.0,
                latent_noise=False,
                noise_seed=1234,
            )
            val_clean = evaluate_reconstruction(
                model,
                val_loader,
                device,
                variant="clean",
                noise_std=0.0,
                latent_noise=False,
                noise_seed=1234,
            )
            val_input_noisy = evaluate_reconstruction(
                model,
                val_loader,
                device,
                variant="noisy",
                noise_std=eval_noise_std,
                latent_noise=False,
                noise_seed=1234,
            )

            scheduler.step(val_clean.loss)
            current_lr = optimizer.param_groups[0]["lr"]

            metrics_hist["loss_train"].append(train_loss)
            metrics_hist["loss_val"].append(val_clean.loss)
            metrics_hist["mse_train"].append(train_eval.mse)
            metrics_hist["psnr_train"].append(train_eval.psnr)
            metrics_hist["ssim_train"].append(train_eval.ssim)
            metrics_hist["mse_val"].append(val_clean.mse)
            metrics_hist["psnr_val"].append(val_clean.psnr)
            metrics_hist["ssim_val"].append(val_clean.ssim)

            csv_writer.writerow([
                epoch,
                train_loss,
                train_eval.mse,
                train_eval.psnr,
                train_eval.ssim,
                val_clean.loss,
                val_clean.mse,
                val_clean.psnr,
                val_clean.ssim,
                val_input_noisy.loss,
                val_input_noisy.mse,
                val_input_noisy.psnr,
                val_input_noisy.ssim,
                disc_loss,
            ])
            csv_f.flush()

            if val_clean.loss < best_val_clean:
                best_val_clean = val_clean.loss
                best_epoch = epoch
                state_dict_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                safe_save_state_dict(state_dict_cpu, ckpt_path)

            if log:
                logger.info(
                    f"Epoch {epoch}/{epochs} | "
                    f"train_loss={train_loss:.6f} | "
                    f"disc_loss={disc_loss:.6f} | "
                    f"val_clean_loss={val_clean.loss:.6f} | "
                    f"val_clean_psnr={val_clean.psnr:.2f} | "
                    f"val_clean_ssim={val_clean.ssim:.4f} | "
                    f"val_input_noisy_loss={val_input_noisy.loss:.6f} | "
                    f"val_input_noisy_psnr={val_input_noisy.psnr:.2f} | "
                    f"val_input_noisy_ssim={val_input_noisy.ssim:.4f} | "
                    f"lr={current_lr:.6e}"
                )

            if log_wandb:
                wandb.log({
                    "epoch": epoch,
                    "lr": current_lr,
                    "train/loss": train_loss,
                    "train/disc_loss": disc_loss,
                    "train_clean/mse": train_eval.mse,
                    "train_clean/psnr": train_eval.psnr,
                    "train_clean/ssim": train_eval.ssim,
                    "val_clean/loss": val_clean.loss,
                    "val_clean/mse": val_clean.mse,
                    "val_clean/psnr": val_clean.psnr,
                    "val_clean/ssim": val_clean.ssim,
                    "val_input_noisy/loss": val_input_noisy.loss,
                    "val_input_noisy/mse": val_input_noisy.mse,
                    "val_input_noisy/psnr": val_input_noisy.psnr,
                    "val_input_noisy/ssim": val_input_noisy.ssim,
                    "best/val_clean_loss": best_val_clean,
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
                latent_noise=False,
                noise_std=0.0,
            )

            if log_wandb:
                wandb.log({
                    "val/reconstructions": wandb.Image(
                        image_path,
                        caption=f"{model_name} | clean | epoch {epoch}",
                    )
                })

            if es.step(val_clean.loss):
                if log:
                    logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

    finally:
        csv_f.close()
        if run is not None:
            run.finish()

    return ckpt_path