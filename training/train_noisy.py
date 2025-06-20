import os
import json
import time
import torch
import random
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchmetrics import MeanSquaredError
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

from data.data_setter import get_subnet_datasets, get_imagenet_datasets
from utils.helpers import get_device, save_images, plot_metrics

def evaluate(loader, model, device):
    mse_metric = MeanSquaredError().to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            x_hat = model(x)
            if isinstance(x_hat, tuple):
                x_hat = x_hat[0]
            x_hat = x_hat.clamp(0, 1)
            mse_metric.update(x_hat, x)
            psnr_metric.update(x_hat, x)
            ssim_metric.update(x_hat, x)

    return (
        mse_metric.compute().item(),
        psnr_metric.compute().item(),
        ssim_metric.compute().item()
    )

def train_model(model_class, config_path, input_variant="noisy", dataset_variant="subset", log=False):
    with open(config_path, "r") as f:
        config = json.load(f)

    device = get_device()
    noise_std = config.get("noise_std", 0.1)

    if dataset_variant == "subset":
        train_set, val_set = get_subnet_datasets("datasets/subset_imagenet/", image_size=config["image_size"])
    else:
        train_set, val_set = get_imagenet_datasets("datasets/full_imagenet/", image_size=config["image_size"])

    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    patience = config.get("early_stopping_patience", 5)
    val_fraction = config.get("val_subset_fraction", 0.1)
    best_val_mse = float("inf")
    epochs_no_improve = 0

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)

    model = model_class(config).to(device)

    is_adversarial = hasattr(model_class, "discriminator_class") and model_class.discriminator_class is not None

    if is_adversarial:
        disc_class = model_class.discriminator_class
        if "latent_dim" in disc_class.__init__.__code__.co_varnames:
            discriminator = disc_class(latent_dim=config["latent_dim"]).to(device)
            disc_input = "latent"
        else:
            discriminator = disc_class().to(device)
            disc_input = "image"

        optimizer_G = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        criterion_recon = nn.MSELoss()
        criterion_adv = nn.BCELoss()
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        criterion = nn.MSELoss()

    suffix = f"_{input_variant}"
    result_dir = os.path.join("results", config["name"] + suffix, "training")
    os.makedirs(os.path.join(result_dir, "images"), exist_ok=True)

    history_keys = ["mse_train", "mse_val", "psnr_train", "psnr_val", "ssim_train", "ssim_val"]
    if is_adversarial:
        history_keys += ["loss_G", "loss_D"]
    if hasattr(model, "get_vq_losses"):
        example_losses = model.get_vq_losses()
        history_keys += list(example_losses.keys())

    history = {key: [] for key in history_keys}
    metrics_path = os.path.join(result_dir, "metrics.txt")

    with open(metrics_path, "w") as f:
        f.write("\t".join(history_keys) + "\n")

    pretrained_path_G = os.path.join("checkpoints", config["name"] + suffix + ".pth")
    pretrained_path_D = os.path.join("checkpoints", config["name"] + suffix + "_discriminator.pth")
    os.makedirs(os.path.dirname(pretrained_path_G), exist_ok=True)
    os.makedirs(os.path.dirname(pretrained_path_D), exist_ok=True)

    for epoch in range(config["epochs"]):
        model.train()
        if is_adversarial:
            discriminator.train()
        print(f"\n--- Epoch {epoch + 1}/{config['epochs']} ---")
        epoch_start = time.time()

        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)
            noise = torch.randn_like(x) * noise_std
            x_noisy = (x + noise).clamp(0.0, 1.0)

            if is_adversarial:
                valid = torch.ones((x.size(0), 1), device=device)
                fake = torch.zeros((x.size(0), 1), device=device)

                optimizer_G.zero_grad()
                x_hat_out = model(x_noisy)
                x_hat = x_hat_out[0] if isinstance(x_hat_out, tuple) else x_hat_out
                adv_input = x_hat if disc_input == "image" else model.encode(x_hat)
                g_loss = criterion_recon(x_hat, x) + criterion_adv(discriminator(adv_input), valid)
                g_loss.backward()
                optimizer_G.step()

                optimizer_D.zero_grad()
                if disc_input == "image":
                    real_input = x
                    fake_input = x_hat.detach()
                else:
                    real_input = model.encode(x).detach()
                    fake_input = model.encode(x_hat).detach()
                real_loss = criterion_adv(discriminator(real_input), valid)
                fake_loss = criterion_adv(discriminator(fake_input), fake)
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                optimizer_D.step()
            else:
                optimizer.zero_grad()
                x_hat_out = model(x_noisy)
                x_hat = x_hat_out[0] if isinstance(x_hat_out, tuple) else x_hat_out
                loss = criterion(x_hat, x)
                loss.backward()
                optimizer.step()

            if batch_idx % max(1, len(train_loader) // 10) == 0:
                elapsed = time.time() - epoch_start
                speed = (batch_idx + 1) / elapsed
                log_str = f"[Epoch {epoch+1}/{config['epochs']}] [Batch {batch_idx}/{len(train_loader)}] Speed: {speed:.2f} it/s"

                recon_loss = nn.functional.mse_loss(x_hat, x)
                log_str += f" loss={recon_loss.item():.6f}"

                if is_adversarial:
                    log_str += f", G_loss={g_loss.item():.6f}, D_loss={d_loss.item():.6f}"
                if hasattr(model, "get_vq_losses"):
                    vq_losses = model.get_vq_losses()
                    for k, v in vq_losses.items():
                        log_str += f", {k}={v:.6f}"
                print(log_str)

        train_mse, train_psnr, train_ssim = evaluate(train_loader, model, device)
        val_subset = Subset(val_set, random.sample(range(len(val_set)), int(val_fraction * len(val_set))))
        val_mse, val_psnr, val_ssim = evaluate(DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers), model, device)

        val_loader_preview = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=num_workers)
        save_images(model, val_loader_preview, device, os.path.join(result_dir, "images", f"epoch_{epoch+1}.png"))

        history["mse_train"].append(train_mse)
        history["mse_val"].append(val_mse)
        history["psnr_train"].append(train_psnr)
        history["psnr_val"].append(val_psnr)
        history["ssim_train"].append(train_ssim)
        history["ssim_val"].append(val_ssim)
        if is_adversarial:
            history["loss_G"].append(g_loss.item())
            history["loss_D"].append(d_loss.item())
        if hasattr(model, "get_vq_losses"):
            vq_losses = model.get_vq_losses()
            for k, v in vq_losses.items():
                if k in history:
                    history[k].append(v)

        with open(metrics_path, "a") as f:
            row = [str(history[k][-1]) for k in history_keys]
            f.write("\t".join(row) + "\n")

        print("[Metrics]", f"Train MSE: {train_mse:.6f}", f"Train PSNR: {train_psnr:.6f}", f"Train SSIM: {train_ssim:.6f}")
        print("[Metrics]", f"Val MSE: {val_mse:.6f}", f"Val PSNR: {val_psnr:.6f}", f"Val SSIM: {val_ssim:.6f}")

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            torch.save(model.state_dict(), pretrained_path_G)
            if is_adversarial:
                torch.save(discriminator.state_dict(), pretrained_path_D)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("[INFO] Early stopping triggered.")
                break

    plot_metrics(history, result_dir)
