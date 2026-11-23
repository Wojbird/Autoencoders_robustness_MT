import logging
import os

import torch
from torch.utils.data import DataLoader

from data.data_setter import get_imagenet_datasets, get_subnet_datasets
from evaluation.evaluate import evaluate_reconstruction
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
from utils.latent import (
    add_gaussian_latent_noise,
    forward_with_latent_transform,
)


logger = logging.getLogger(__name__)


def _assert_no_vector_quantization(model) -> None:
    if hasattr(model, "vq_loss") or hasattr(model, "vq_losses"):
        raise ValueError(
            "train_noisy_latent_model() does not support VQ-VAE models."
        )


def _forward_noisy_latent(
    model,
    x: torch.Tensor,
    noise_std: float,
    relative: bool = True,
) -> torch.Tensor:
    def transform(z: torch.Tensor) -> torch.Tensor:
        return add_gaussian_latent_noise(
            z,
            noise_std=noise_std,
            relative=relative,
        )

    return forward_with_latent_transform(
        model,
        x,
        transform,
    )


def _state_dict_to_cpu(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in module.state_dict().items()
    }


def train_noisy_latent_model(
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
    learning_rate = float(cfg["learning_rate"])
    weight_decay = float(cfg.get("weight_decay", 0.0))

    noise_std = float(
        cfg.get("noise_latent", cfg.get("noise_std", 0.0))
    )
    latent_noise_relative = bool(
        cfg.get("latent_noise_relative", True)
    )
    eval_input_noise_std = float(
        cfg.get("eval_noise_std", cfg.get("noise_std", 0.0))
    )

    if noise_std < 0.0:
        raise ValueError("noise_latent must be non-negative.")
    if eval_input_noise_std < 0.0:
        raise ValueError("eval_noise_std must be non-negative.")

    val_fraction = float(cfg.get("val_subset_fraction", 1.0))
    train_eval_fraction = float(cfg.get("train_eval_fraction", 0.1))
    patience = int(cfg.get("early_stopping_patience", 5))
    scheduler_factor = float(cfg.get("scheduler_factor", 0.5))
    scheduler_patience = int(cfg.get("scheduler_patience", 5))
    scheduler_min_lr = float(cfg.get("scheduler_min_lr", 1e-6))
    scheduler_threshold = float(cfg.get("scheduler_threshold", 1e-4))
    experiment_seed = int(cfg.get("experiment_seed", 42))
    dataset_split_seed = int(cfg.get("dataset_split_seed", 42))
    eval_noise_seed = int(cfg.get("eval_noise_seed", 1234))
    dataloader_seed = int(cfg.get("dataloader_seed", 42))

    if scheduler_patience >= patience:
        raise ValueError(
            "scheduler_patience must be smaller than "
            "early_stopping_patience."
        )

    set_seed(experiment_seed)
    device = get_device(gpu_id)
    pin_memory = device.type == "cuda"

    model = model_class(cfg).to(device)
    _assert_no_vector_quantization(model)

    discriminator, optimizer_d, bce, adv_weight = build_discriminator(
        model,
        cfg,
        device,
    )

    if dataset_type == "full":
        train_set, val_set = get_imagenet_datasets(
            image_size=image_size,
        )
    elif dataset_type == "subset":
        train_set, val_set = get_subnet_datasets(
            image_size=image_size,
            val_split=0.1,
            split_seed=dataset_split_seed,
        )
    else:
        raise ValueError("dataset_type must be 'subset' or 'full'.")

    val_set = ensure_val_fraction(
        val_set,
        val_fraction,
        split_seed=dataset_split_seed + 1,
    )
    train_eval_set = ensure_val_fraction(
        train_set,
        train_eval_fraction,
        split_seed=dataloader_seed + 1,
    )

    train_generator = torch.Generator().manual_seed(dataloader_seed)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=train_generator,
        persistent_workers=num_workers > 0,
    )
    train_eval_loader = DataLoader(
        train_eval_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    reconstruction_loss_fn = torch.nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience,
        threshold=scheduler_threshold,
        min_lr=scheduler_min_lr,
    )

    results_dir = os.path.join(
        "results",
        model_name,
        dataset_type,
        "noisy_latent",
    )
    checkpoint_dir = os.path.join(
        "checkpoints",
        model_name,
        dataset_type,
        "noisy_latent",
    )
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"{model_name}_noisy_latent_best.pt",
    )
    discriminator_checkpoint_path = os.path.join(
        checkpoint_dir,
        f"{model_name}_noisy_latent_discriminator_best.pt",
    )
    csv_path = os.path.join(results_dir, "metrics_per_epoch.csv")

    csv_file, csv_writer = init_csv_logger(
        csv_path,
        extra_columns=[
            "val_clean_loss",
            "val_clean_mse",
            "val_clean_psnr",
            "val_clean_ssim",
            "val_input_noisy_loss",
            "val_input_noisy_mse",
            "val_input_noisy_psnr",
            "val_input_noisy_ssim",
            "train_recon_loss",
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

    early_stopping = EarlyStopping(
        patience=patience,
        min_delta=0.0,
    )
    best_val_latent_noisy = float("inf")
    best_epoch = 0

    wandb_module = None
    run = None

    if log_wandb:
        try:
            import wandb as wandb_module
        except ImportError as exc:
            raise RuntimeError(
                "W&B logging was requested, but the 'wandb' package "
                "is not installed."
            ) from exc

        run = wandb_module.init(
            settings=wandb_module.Settings(
                init_timeout=180,
                start_method="thread",
            ),
            project="autoencoders-robustness",
            name=f"{model_name}_{dataset_type}_noisy_latent",
            config={
                **cfg,
                "dataset_type": dataset_type,
                "training_variant": "noisy_latent",
                "model_class": model_class.__name__,
                "gpu_id": gpu_id,
                "selection_metric": "val_latent_noisy_loss",
                "latent_noise_relative": latent_noise_relative,
            },
        )

    try:
        for epoch in range(1, epochs + 1):
            model.train()

            if discriminator is not None:
                discriminator.train()

            train_total_loss_sum = 0.0
            train_recon_loss_sum = 0.0
            train_adv_loss_sum = 0.0
            discriminator_loss_sum = 0.0
            number_of_samples = 0

            for x, _ in train_loader:
                x = x.to(
                    device,
                    non_blocking=pin_memory,
                )
                current_batch_size = x.shape[0]

                optimizer.zero_grad(set_to_none=True)

                # One reconstruction is shared by the discriminator and
                # generator steps, so both see the same latent-noise sample.
                x_hat = _forward_noisy_latent(
                    model,
                    x,
                    noise_std=noise_std,
                    relative=latent_noise_relative,
                )
                x_hat = torch.clamp(x_hat, 0.0, 1.0)

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

                reconstruction_loss = reconstruction_loss_fn(x_hat, x)

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
                total_loss = reconstruction_loss + adversarial_contribution

                total_loss.backward()
                optimizer.step()

                train_total_loss_sum += (
                    float(total_loss.item()) * current_batch_size
                )
                train_recon_loss_sum += (
                    float(reconstruction_loss.item()) * current_batch_size
                )
                train_adv_loss_sum += (
                    float(adversarial_loss.item()) * current_batch_size
                )
                number_of_samples += current_batch_size

            if number_of_samples == 0:
                raise ValueError("The training dataloader is empty.")

            train_total_loss = train_total_loss_sum / number_of_samples
            train_recon_loss = train_recon_loss_sum / number_of_samples
            train_adv_loss = train_adv_loss_sum / number_of_samples
            train_adv_contribution = adv_weight * train_adv_loss

            discriminator_loss = (
                discriminator_loss_sum / number_of_samples
                if discriminator is not None
                else 0.0
            )

            train_eval_latent_noisy = evaluate_reconstruction(
                model,
                train_eval_loader,
                device,
                variant="noisy_latent",
                noise_std=noise_std,
                latent_noise=True,
                noise_seed=eval_noise_seed,
                latent_noise_relative=latent_noise_relative,
            )

            val_latent_noisy = evaluate_reconstruction(
                model,
                val_loader,
                device,
                variant="noisy_latent",
                noise_std=noise_std,
                latent_noise=True,
                noise_seed=eval_noise_seed,
                latent_noise_relative=latent_noise_relative,
            )

            val_clean = evaluate_reconstruction(
                model,
                val_loader,
                device,
                variant="clean",
                noise_std=0.0,
                latent_noise=False,
                noise_seed=eval_noise_seed,
                latent_noise_relative=latent_noise_relative,
            )

            val_input_noisy = evaluate_reconstruction(
                model,
                val_loader,
                device,
                variant="noisy",
                noise_std=eval_input_noise_std,
                latent_noise=False,
                noise_seed=eval_noise_seed,
                latent_noise_relative=latent_noise_relative,
            )

            scheduler.step(val_latent_noisy.loss)
            current_learning_rate = optimizer.param_groups[0]["lr"]

            metrics_history["loss_train"].append(train_recon_loss)
            metrics_history["loss_val"].append(val_latent_noisy.loss)
            metrics_history["mse_train"].append(train_eval_latent_noisy.mse)
            metrics_history["psnr_train"].append(train_eval_latent_noisy.psnr)
            metrics_history["ssim_train"].append(train_eval_latent_noisy.ssim)
            metrics_history["mse_val"].append(val_latent_noisy.mse)
            metrics_history["psnr_val"].append(val_latent_noisy.psnr)
            metrics_history["ssim_val"].append(val_latent_noisy.ssim)

            csv_writer.writerow([
                epoch,
                train_total_loss,
                train_eval_latent_noisy.mse,
                train_eval_latent_noisy.psnr,
                train_eval_latent_noisy.ssim,
                val_latent_noisy.loss,
                val_latent_noisy.mse,
                val_latent_noisy.psnr,
                val_latent_noisy.ssim,
                val_clean.loss,
                val_clean.mse,
                val_clean.psnr,
                val_clean.ssim,
                val_input_noisy.loss,
                val_input_noisy.mse,
                val_input_noisy.psnr,
                val_input_noisy.ssim,
                train_recon_loss,
                train_adv_loss,
                train_adv_contribution,
                discriminator_loss,
            ])
            csv_file.flush()

            if val_latent_noisy.loss < best_val_latent_noisy:
                best_val_latent_noisy = val_latent_noisy.loss
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
                    f"Epoch {epoch}/{epochs} | "
                    f"train_total_loss={train_total_loss:.6f} | "
                    f"train_recon_loss={train_recon_loss:.6f} | "
                    f"train_adv_loss={train_adv_loss:.6f} | "
                    f"disc_loss={discriminator_loss:.6f} | "
                    f"val_latent_noisy_loss={val_latent_noisy.loss:.6f} | "
                    f"val_latent_noisy_psnr={val_latent_noisy.psnr:.2f} | "
                    f"val_latent_noisy_ssim={val_latent_noisy.ssim:.4f} | "
                    f"val_clean_loss={val_clean.loss:.6f} | "
                    f"val_input_noisy_loss={val_input_noisy.loss:.6f} | "
                    f"lr={current_learning_rate:.6e} | "
                    f"latent_noise_std={noise_std:.6f}"
                )

            if run is not None:
                run.log({
                    "epoch": epoch,
                    "lr": current_learning_rate,
                    "latent_noise_std": noise_std,
                    "train/loss": train_total_loss,
                    "train/total_loss": train_total_loss,
                    "train/recon_loss": train_recon_loss,
                    "train/adv_loss": train_adv_loss,
                    "train/adv_contribution": train_adv_contribution,
                    "train/disc_loss": discriminator_loss,
                    "train_latent_noisy/mse": train_eval_latent_noisy.mse,
                    "train_latent_noisy/psnr": train_eval_latent_noisy.psnr,
                    "train_latent_noisy/ssim": train_eval_latent_noisy.ssim,
                    "val_latent_noisy/loss": val_latent_noisy.loss,
                    "val_latent_noisy/mse": val_latent_noisy.mse,
                    "val_latent_noisy/psnr": val_latent_noisy.psnr,
                    "val_latent_noisy/ssim": val_latent_noisy.ssim,
                    "val_clean/loss": val_clean.loss,
                    "val_clean/mse": val_clean.mse,
                    "val_clean/psnr": val_clean.psnr,
                    "val_clean/ssim": val_clean.ssim,
                    "val_input_noisy/loss": val_input_noisy.loss,
                    "val_input_noisy/mse": val_input_noisy.mse,
                    "val_input_noisy/psnr": val_input_noisy.psnr,
                    "val_input_noisy/ssim": val_input_noisy.ssim,
                    "best/val_latent_noisy_loss": best_val_latent_noisy,
                    "best/epoch": best_epoch,
                })

            plot_metrics(metrics_history, results_dir)

            images_dir = os.path.join(results_dir, "images")
            os.makedirs(images_dir, exist_ok=True)

            image_path = os.path.join(
                images_dir,
                f"recon_epoch_{epoch:04d}.png",
            )
            save_images(
                model=model,
                dataloader=val_loader,
                device=device,
                save_path=image_path,
                num_images=8,
                add_noise=False,
                latent_noise=True,
                noise_std=noise_std,
                latent_noise_relative=latent_noise_relative,
            )

            if run is not None:
                run.log({
                    "val/reconstructions": wandb_module.Image(
                        image_path,
                        caption=(
                            f"{model_name} | noisy_latent | epoch {epoch}"
                        ),
                    )
                })

            if early_stopping.step(val_latent_noisy.loss):
                if log:
                    logger.info(
                        f"Early stopping triggered at epoch {epoch}."
                    )
                break

    finally:
        csv_file.close()

        if run is not None:
            run.finish()

    return checkpoint_path
