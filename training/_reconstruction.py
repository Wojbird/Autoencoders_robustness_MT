from __future__ import annotations

import logging
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.data_setter import get_imagenet_datasets, get_subnet_datasets
from evaluation.evaluate import compute_training_loss, evaluate_reconstruction
from training._adversarial import (
    build_discriminator,
    discriminator_step,
    generator_adv_loss,
)
from utils.helpers import (
    EarlyStopping,
    ensure_val_fraction,
    get_device,
    init_csv_logger,
    load_config,
    plot_metrics,
    safe_save_state_dict,
    save_images,
    set_seed,
)
from utils.latent import unwrap_tensor


logger = logging.getLogger(__name__)

SUPPORTED_VARIANTS = frozenset({"clean", "noisy"})


def _state_dict_to_cpu(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in module.state_dict().items()
    }


def _validate_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, received {value}.")


def _build_datasets(
    *,
    dataset_type: str,
    image_size: int,
    split_seed: int,
):
    if dataset_type == "full":
        return get_imagenet_datasets(image_size=image_size)
    if dataset_type == "subset":
        return get_subnet_datasets(
            image_size=image_size,
            val_split=0.1,
            split_seed=split_seed,
        )
    raise ValueError("dataset_type must be 'subset' or 'full'.")


def _build_loaders(
    *,
    train_set,
    val_set,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    dataloader_seed: int,
    train_eval_fraction: float,
):
    train_eval_set = ensure_val_fraction(
        train_set,
        train_eval_fraction,
        split_seed=dataloader_seed + 1,
    )
    train_generator = torch.Generator().manual_seed(dataloader_seed)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        generator=train_generator,
        **common,
    )
    train_eval_loader = DataLoader(
        train_eval_set,
        shuffle=False,
        **common,
    )
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        **common,
    )
    return train_loader, train_eval_loader, val_loader


def _init_wandb(
    *,
    enabled: bool,
    model_name: str,
    dataset_type: str,
    variant: str,
    config: dict,
    model_class,
    gpu_id: int | None,
    selection_metric: str,
):
    if not enabled:
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B logging was requested, but the 'wandb' package is not installed."
        ) from exc

    return wandb.init(
        settings=wandb.Settings(init_timeout=180, start_method="thread"),
        project="autoencoders-robustness",
        name=f"{model_name}_{dataset_type}_{variant}",
        config={
            **config,
            "dataset_type": dataset_type,
            "training_variant": variant,
            "model_class": model_class.__name__,
            "gpu_id": gpu_id,
            "selection_metric": selection_metric,
        },
    )


def train_reconstruction_model(
    model_class,
    config_path: str,
    *,
    variant: str,
    dataset_type: str = "subset",
    log: bool = False,
    gpu_id: int | None = None,
    log_wandb: bool = False,
) -> str:
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"variant must be one of {sorted(SUPPORTED_VARIANTS)}.")

    cfg = load_config(config_path)
    model_name = str(cfg["name"])
    image_size = int(cfg["image_size"])
    batch_size = int(cfg["batch_size"])
    num_workers = int(cfg.get("num_workers", 4))
    epochs = int(cfg["epochs"])
    learning_rate = float(cfg["learning_rate"])
    weight_decay = float(cfg.get("weight_decay", 0.0))
    noise_std = float(cfg.get("noise_std", 0.0))
    eval_noise_std = float(cfg.get("eval_noise_std", noise_std))
    val_fraction = float(cfg.get("val_subset_fraction", 1.0))
    train_eval_fraction = float(cfg.get("train_eval_fraction", 0.1))
    patience = int(cfg.get("early_stopping_patience", 5))
    scheduler_factor = float(cfg.get("scheduler_factor", 0.5))
    scheduler_patience = int(cfg.get("scheduler_patience", 2))
    scheduler_min_lr = float(cfg.get("scheduler_min_lr", 1e-6))
    scheduler_threshold = float(cfg.get("scheduler_threshold", 1e-4))
    experiment_seed = int(cfg.get("experiment_seed", 42))
    dataset_split_seed = int(cfg.get("dataset_split_seed", 42))
    dataloader_seed = int(cfg.get("dataloader_seed", 42))
    eval_noise_seed = int(cfg.get("eval_noise_seed", 1234))

    _validate_positive("image_size", image_size)
    _validate_positive("batch_size", batch_size)
    _validate_positive("epochs", epochs)
    _validate_positive("learning_rate", learning_rate)
    _validate_positive("early_stopping_patience", patience)
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative.")
    if weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative.")
    if noise_std < 0.0 or eval_noise_std < 0.0:
        raise ValueError("Noise standard deviations must be non-negative.")
    if scheduler_patience >= patience:
        raise ValueError(
            "scheduler_patience must be smaller than early_stopping_patience "
            "so the scheduler can act before training stops."
        )

    set_seed(experiment_seed)
    device = get_device(gpu_id)
    pin_memory = device.type == "cuda"

    model = model_class(cfg).to(device)
    discriminator, optimizer_d, bce, adv_weight = build_discriminator(
        model,
        cfg,
        device,
    )

    train_set, val_set = _build_datasets(
        dataset_type=dataset_type,
        image_size=image_size,
        split_seed=dataset_split_seed,
    )
    val_set = ensure_val_fraction(
        val_set,
        val_fraction,
        split_seed=dataset_split_seed + 1,
    )
    train_loader, train_eval_loader, val_loader = _build_loaders(
        train_set=train_set,
        val_set=val_set,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        dataloader_seed=dataloader_seed,
        train_eval_fraction=train_eval_fraction,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience,
        threshold=scheduler_threshold,
        min_lr=scheduler_min_lr,
    )

    results_dir = os.path.join("results", model_name, dataset_type, variant)
    checkpoint_dir = os.path.join("checkpoints", model_name, dataset_type, variant)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"{model_name}_{variant}_best.pt",
    )
    discriminator_checkpoint_path = os.path.join(
        checkpoint_dir,
        f"{model_name}_{variant}_discriminator_best.pt",
    )
    csv_path = os.path.join(results_dir, "metrics_per_epoch.csv")

    secondary_prefix = "val_input_noisy" if variant == "clean" else "val_clean"
    csv_file, csv_writer = init_csv_logger(
        csv_path,
        extra_columns=[
            f"{secondary_prefix}_loss",
            f"{secondary_prefix}_mse",
            f"{secondary_prefix}_psnr",
            f"{secondary_prefix}_ssim",
            "train_recon_loss",
            "train_vq_loss",
            "train_adv_loss",
            "train_adv_contribution",
            "disc_loss",
        ],
        overwrite=True,
    )

    metrics_history = {
        "mse_train": [],
        "psnr_train": [],
        "ssim_train": [],
        "mse_val": [],
        "psnr_val": [],
        "ssim_val": [],
        "loss_train": [],
        "loss_val": [],
    }
    early_stopping = EarlyStopping(patience=patience, min_delta=0.0)
    best_validation_loss = float("inf")
    best_epoch = 0
    selection_metric = "val_clean_loss" if variant == "clean" else "val_input_noisy_loss"

    run = _init_wandb(
        enabled=log_wandb,
        model_name=model_name,
        dataset_type=dataset_type,
        variant=variant,
        config=cfg,
        model_class=model_class,
        gpu_id=gpu_id,
        selection_metric=selection_metric,
    )

    try:
        for epoch in range(1, epochs + 1):
            model.train()
            if discriminator is not None:
                discriminator.train()

            total_loss_sum = 0.0
            reconstruction_loss_sum = 0.0
            vq_loss_sum = 0.0
            adversarial_loss_sum = 0.0
            discriminator_loss_sum = 0.0
            number_of_samples = 0

            for x, _ in train_loader:
                x = x.to(device, non_blocking=pin_memory)
                current_batch_size = x.shape[0]

                if variant == "noisy":
                    model_input = torch.clamp(
                        x + noise_std * torch.randn_like(x),
                        0.0,
                        1.0,
                    )
                else:
                    model_input = x

                optimizer.zero_grad(set_to_none=True)
                x_hat = unwrap_tensor(model(model_input))

                # The discriminator and generator use this same reconstruction.
                # This is important for stochastic models and dropout-enabled paths.
                if discriminator is not None:
                    discriminator_loss_value = discriminator_step(
                        discriminator=discriminator,
                        optimizer_d=optimizer_d,
                        bce=bce,
                        x_real=x,
                        x_fake=x_hat,
                    )
                    discriminator_loss_sum += (
                        discriminator_loss_value * current_batch_size
                    )

                reconstruction_loss = F.mse_loss(x_hat, x)
                reconstruction_and_vq_loss = compute_training_loss(
                    model,
                    x_hat,
                    x,
                    allow_vq=True,
                )
                vq_loss = reconstruction_and_vq_loss - reconstruction_loss

                if discriminator is not None:
                    adversarial_loss = generator_adv_loss(
                        discriminator,
                        bce,
                        x_hat,
                    )
                else:
                    adversarial_loss = torch.zeros(
                        (),
                        device=device,
                        dtype=reconstruction_loss.dtype,
                    )

                adversarial_contribution = adv_weight * adversarial_loss
                total_loss = reconstruction_and_vq_loss + adversarial_contribution
                total_loss.backward()
                optimizer.step()

                total_loss_sum += float(total_loss.item()) * current_batch_size
                reconstruction_loss_sum += (
                    float(reconstruction_loss.item()) * current_batch_size
                )
                vq_loss_sum += float(vq_loss.item()) * current_batch_size
                adversarial_loss_sum += (
                    float(adversarial_loss.item()) * current_batch_size
                )
                number_of_samples += current_batch_size

            if number_of_samples == 0:
                raise ValueError("The training dataloader is empty.")

            train_total_loss = total_loss_sum / number_of_samples
            train_reconstruction_loss = reconstruction_loss_sum / number_of_samples
            train_vq_loss = vq_loss_sum / number_of_samples
            train_adversarial_loss = adversarial_loss_sum / number_of_samples
            train_adversarial_contribution = adv_weight * train_adversarial_loss
            train_discriminator_loss = (
                discriminator_loss_sum / number_of_samples
                if discriminator is not None
                else 0.0
            )

            primary_eval = evaluate_reconstruction(
                model,
                train_eval_loader,
                device,
                variant=variant,
                noise_std=noise_std if variant == "noisy" else 0.0,
                latent_noise=False,
                noise_seed=eval_noise_seed,
            )
            validation_primary = evaluate_reconstruction(
                model,
                val_loader,
                device,
                variant=variant,
                noise_std=noise_std if variant == "noisy" else 0.0,
                latent_noise=False,
                noise_seed=eval_noise_seed,
            )
            secondary_variant = "noisy" if variant == "clean" else "clean"
            validation_secondary = evaluate_reconstruction(
                model,
                val_loader,
                device,
                variant=secondary_variant,
                noise_std=eval_noise_std if secondary_variant == "noisy" else 0.0,
                latent_noise=False,
                noise_seed=eval_noise_seed,
            )

            scheduler.step(validation_primary.loss)
            current_learning_rate = optimizer.param_groups[0]["lr"]

            metrics_history["loss_train"].append(train_reconstruction_loss)
            metrics_history["loss_val"].append(validation_primary.loss)
            metrics_history["mse_train"].append(primary_eval.mse)
            metrics_history["psnr_train"].append(primary_eval.psnr)
            metrics_history["ssim_train"].append(primary_eval.ssim)
            metrics_history["mse_val"].append(validation_primary.mse)
            metrics_history["psnr_val"].append(validation_primary.psnr)
            metrics_history["ssim_val"].append(validation_primary.ssim)

            csv_writer.writerow([
                epoch,
                train_total_loss,
                primary_eval.mse,
                primary_eval.psnr,
                primary_eval.ssim,
                validation_primary.loss,
                validation_primary.mse,
                validation_primary.psnr,
                validation_primary.ssim,
                validation_secondary.loss,
                validation_secondary.mse,
                validation_secondary.psnr,
                validation_secondary.ssim,
                train_reconstruction_loss,
                train_vq_loss,
                train_adversarial_loss,
                train_adversarial_contribution,
                train_discriminator_loss,
            ])
            csv_file.flush()

            if validation_primary.loss < best_validation_loss:
                best_validation_loss = validation_primary.loss
                best_epoch = epoch
                safe_save_state_dict(
                    _state_dict_to_cpu(model),
                    checkpoint_path,
                )
                if discriminator is not None:
                    safe_save_state_dict(
                        _state_dict_to_cpu(discriminator),
                        discriminator_checkpoint_path,
                    )

            if log:
                logger.info(
                    "Epoch %d/%d | train_total=%.6f | train_recon=%.6f | "
                    "train_vq=%.6f | train_adv=%.6f | disc=%.6f | "
                    "val_primary=%.6f | val_psnr=%.2f | val_ssim=%.4f | "
                    "val_secondary=%.6f | lr=%.6e",
                    epoch,
                    epochs,
                    train_total_loss,
                    train_reconstruction_loss,
                    train_vq_loss,
                    train_adversarial_loss,
                    train_discriminator_loss,
                    validation_primary.loss,
                    validation_primary.psnr,
                    validation_primary.ssim,
                    validation_secondary.loss,
                    current_learning_rate,
                )

            if run is not None:
                run.log({
                    "epoch": epoch,
                    "lr": current_learning_rate,
                    "train/total_loss": train_total_loss,
                    "train/recon_loss": train_reconstruction_loss,
                    "train/vq_loss": train_vq_loss,
                    "train/adv_loss": train_adversarial_loss,
                    "train/adv_contribution": train_adversarial_contribution,
                    "train/disc_loss": train_discriminator_loss,
                    f"train_{variant}/mse": primary_eval.mse,
                    f"train_{variant}/psnr": primary_eval.psnr,
                    f"train_{variant}/ssim": primary_eval.ssim,
                    f"val_{variant}/loss": validation_primary.loss,
                    f"val_{variant}/mse": validation_primary.mse,
                    f"val_{variant}/psnr": validation_primary.psnr,
                    f"val_{variant}/ssim": validation_primary.ssim,
                    f"val_{secondary_variant}/loss": validation_secondary.loss,
                    f"val_{secondary_variant}/mse": validation_secondary.mse,
                    f"val_{secondary_variant}/psnr": validation_secondary.psnr,
                    f"val_{secondary_variant}/ssim": validation_secondary.ssim,
                    "best/validation_loss": best_validation_loss,
                    "best/epoch": best_epoch,
                })

            plot_metrics(metrics_history, results_dir)
            images_dir = os.path.join(results_dir, "images")
            os.makedirs(images_dir, exist_ok=True)
            image_path = os.path.join(images_dir, f"recon_epoch_{epoch:04d}.png")
            save_images(
                model=model,
                dataloader=val_loader,
                device=device,
                save_path=image_path,
                num_images=8,
                add_noise=variant == "noisy",
                latent_noise=False,
                noise_std=noise_std if variant == "noisy" else 0.0,
                noise_seed=eval_noise_seed,
            )

            if run is not None:
                import wandb

                run.log({
                    "val/reconstructions": wandb.Image(
                        image_path,
                        caption=f"{model_name} | {variant} | epoch {epoch}",
                    )
                })

            if early_stopping.step(validation_primary.loss):
                if log:
                    logger.info("Early stopping triggered at epoch %d.", epoch)
                break

    finally:
        csv_file.close()
        if run is not None:
            run.finish()

    return checkpoint_path
