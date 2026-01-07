import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
logger = logging.getLogger(__name__)

from data.data_setter import get_subnet_datasets, get_imagenet_datasets
from evaluation.evaluate import evaluate_reconstruction, compute_training_loss
from utils.helpers import get_device, save_images, plot_metrics, load_config, ensure_val_fraction, make_results_dir, init_csv_logger, EarlyStopping


def train_noisy_model(
    model_class,
    config_path: str,
    *,
    dataset_type: str = "subset",
    log: bool = False,
) -> str:
    """
    Denoising AE training: input is noisy, target is clean.
    Supports AE + VQ-VAE + VQ-VAE2.
    """
    cfg = load_config(config_path)

    model_name = cfg["name"]
    image_size = int(cfg["image_size"])
    batch_size = int(cfg["batch_size"])
    num_workers = int(cfg.get("num_workers", 4))
    epochs = int(cfg["epochs"])
    lr = float(cfg["learning_rate"])
    wd = float(cfg.get("weight_decay", 0.0))
    noise_std = float(cfg.get("noise_std", 0.1))
    val_fraction = float(cfg.get("val_subset_fraction", 1.0))
    patience = int(cfg.get("early_stopping_patience", 10))

    device = get_device()
    model = model_class().to(device)

    if dataset_type == "full":
        train_set, val_set = get_imagenet_datasets(image_size=image_size)
    elif dataset_type == "subset":
        train_set, val_set = get_subnet_datasets(image_size=image_size, val_split=0.1)
        val_set = ensure_val_fraction(val_set, val_fraction)
    else:
        raise ValueError("dataset_type must be 'subset' or 'full'.")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    results_dir = make_results_dir(model_name, dataset_type, "noisy")
    os.makedirs(results_dir, exist_ok=True)

    ckpt_path = os.path.join(results_dir, f"{model_name}_noisy_best.pt")
    csv_path = os.path.join(results_dir, "metrics_per_epoch.csv")
    csv_f, csv_writer = init_csv_logger(csv_path)

    metrics_hist = {
        "mse_train": [], "psnr_train": [], "ssim_train": [],
        "mse_val": [], "psnr_val": [], "ssim_val": [],
        "loss_train": [], "loss_val": [],
    }

    es = EarlyStopping(patience=patience, min_delta=0.0)
    best_val = float("inf")
    loss_fn = nn.MSELoss()

    try:
        for epoch in range(1, epochs + 1):
            model.train()
            train_loss_sum = 0.0
            n_batches = 0

            for x, _ in train_loader:
                x = x.to(device)
                x_in = torch.clamp(x + noise_std * torch.randn_like(x), 0.0, 1.0)

                optimizer.zero_grad(set_to_none=True)
                x_hat = model(x_in)
                x_hat = torch.clamp(x_hat, 0.0, 1.0)

                loss = compute_training_loss(model, x_hat, x, allow_vq=True)
                loss.backward()
                optimizer.step()

                train_loss_sum += float(loss.item())
                n_batches += 1

            train_loss = train_loss_sum / max(1, n_batches)

            # Train metrics (small window)
            train_eval = evaluate_reconstruction(
                model, train_loader, device,
                variant="noisy", noise_std=noise_std, latent_noise=False,
                max_batches=20
            )

            # Validation
            val_eval = evaluate_reconstruction(
                model, val_loader, device,
                variant="noisy", noise_std=noise_std, latent_noise=False
            )

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

            if log:
                logger.info(
                    f"Epoch {epoch}/{epochs} | "
                    f"train_loss={train_loss:.6f} | "
                    f"val_loss={val_eval.loss:.6f} | "
                    f"PSNR={val_eval.psnr:.2f} | "
                    f"SSIM={val_eval.ssim:.4f}"
                )

            save_images(
                model=model,
                dataloader=val_loader,
                device=device,
                save_path=os.path.join(results_dir, f"recon_epoch_{epoch:04d}.png"),
                num_images=8,
                add_noise=True,  # <- noisy input
                latent_noise=False,
                noise_std=noise_std
            )

            if val_eval.loss < best_val:
                best_val = val_eval.loss
                torch.save(model.state_dict(), ckpt_path)

            if es.step(val_eval.loss):
                break

        plot_metrics(metrics_hist, results_dir)

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

    return ckpt_path