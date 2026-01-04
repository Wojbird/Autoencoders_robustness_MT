import os
import json
import time
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchmetrics import MeanSquaredError
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

from data.data_setter import get_subnet_datasets, get_imagenet_datasets
from utils.helpers import get_device, save_images, plot_metrics, set_seed

VARIANT = "noisy_latent"

def is_vq_model(model) -> bool:
    name = model.__class__.__name__.lower()
    return (
        hasattr(model, "vq_loss")
        or hasattr(model, "vq_losses")
        or hasattr(model, "quantizer")
        or hasattr(model, "top_quantizer")
        or hasattr(model, "bottom_quantizer")
        or ("vq" in name)
    )

def evaluate(loader, model, device, noise_std=0.1, seed=None):
    mse_metric = MeanSquaredError().to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=2.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)

    g = None
    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(int(seed))

    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)

            z = model.encode(x)
            if g is None:
                eps = torch.randn_like(z) * noise_std
            else:
                eps = torch.randn(z.shape, device=z.device, generator=g) * noise_std

            x_hat = model.decode(z + eps)
            if isinstance(x_hat, (tuple, list)):
                x_hat = x_hat[0]
            x_hat = x_hat.clamp(-1.0, 1.0)

            mse_metric.update(x_hat, x)
            psnr_metric.update(x_hat, x)
            ssim_metric.update(x_hat, x)

    return mse_metric.compute().item(), psnr_metric.compute().item(), ssim_metric.compute().item()


def train_model(model_class, config_path, dataset_variant="subset", log=False):
    with open(config_path, "r") as f:
        config = json.load(f)

    set_seed(int(config.get("seed", 1337)))
    device = get_device()

    noise_std = float(config.get("noise_std", 0.1))

    if dataset_variant == "subset":
        train_set, val_set = get_subnet_datasets(image_size=config["image_size"])
    else:
        train_set, val_set = get_imagenet_datasets(image_size=config["image_size"])

    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    patience = config.get("early_stopping_patience", 5)
    val_fraction = config.get("val_subset_fraction", 0.1)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)

    val_size = max(1, int(val_fraction * len(val_set)))
    val_indices = random.sample(range(len(val_set)), val_size)
    val_loader = DataLoader(
        Subset(val_set, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    model = model_class(config).to(device)

    if is_vq_model(model):
        print(f"[INFO] Skipping training (noisy_latent): {model.__class__.__name__} is a VQ model.")
        return

    # hard rule: no VQ stuff in noisy_latent trainer
    if hasattr(model, "quantizer") or hasattr(model, "vq_loss") or hasattr(model, "vq_losses") or \
       hasattr(model, "top_quantizer") or hasattr(model, "bottom_quantizer") or \
       ("vq" in model.__class__.__name__.lower()):
        print(f"[INFO] Skipping training: Model {model.__class__.__name__} is not supported in noisy_latent mode.")
        return

    is_adversarial = hasattr(model_class, "discriminator_class") and model_class.discriminator_class is not None

    if is_adversarial:
        disc_class = model_class.discriminator_class
        if "latent_dim" in disc_class.__init__.__code__.co_varnames:
            discriminator = disc_class(latent_dim=config["latent_dim"]).to(device)
            disc_input = "latent"
        else:
            discriminator = disc_class().to(device)
            disc_input = "image"

        optimizer_G = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                       betas=(config["betas_start"], config["betas_end"]),
                                       weight_decay=config["weight_decay"])
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config["learning_rate"],
                                       betas=(config["betas_start"], config["betas_end"]),
                                       weight_decay=config["weight_decay"])
        criterion_recon = nn.MSELoss()
        criterion_adv = nn.BCELoss()
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                     betas=(config["betas_start"], config["betas_end"]),
                                     weight_decay=config["weight_decay"])
        criterion = nn.MSELoss()

    suffix = f"_{VARIANT}"
    result_dir = os.path.join("results", config["name"] + suffix, "training")
    os.makedirs(os.path.join(result_dir, "images"), exist_ok=True)

    history_keys = ["mse_train", "mse_val", "psnr_train", "psnr_val", "ssim_train", "ssim_val"]
    if is_adversarial:
        history_keys += ["loss_G", "loss_D"]

    history = {k: [] for k in history_keys}
    metrics_path = os.path.join(result_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("\t".join(history_keys) + "\n")

    pretrained_path_G = os.path.join("checkpoints", config["name"] + suffix + ".pth")
    pretrained_path_D = os.path.join("checkpoints", config["name"] + suffix + "_discriminator.pth")
    os.makedirs(os.path.dirname(pretrained_path_G), exist_ok=True)
    os.makedirs(os.path.dirname(pretrained_path_D), exist_ok=True)

    best_val_mse = float("inf")
    epochs_no_improve = 0

    for epoch in range(config["epochs"]):
        model.train()
        if is_adversarial:
            discriminator.train()

        print(f"\n--- Epoch {epoch + 1}/{config['epochs']} ---")
        epoch_start = time.time()

        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)

            z = model.encode(x)
            eps = torch.randn_like(z) * noise_std
            x_hat = model.decode(z + eps)
            if isinstance(x_hat, (tuple, list)):
                x_hat = x_hat[0]

            if is_adversarial:
                valid = torch.ones((x.size(0), 1), device=device)
                fake = torch.zeros((x.size(0), 1), device=device)

                optimizer_G.zero_grad()
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
                d_loss = 0.5 * (real_loss + fake_loss)
                d_loss.backward()
                optimizer_D.step()
            else:
                optimizer.zero_grad()
                loss = criterion(x_hat, x)
                loss.backward()
                optimizer.step()

            if batch_idx % max(1, len(train_loader) // 10) == 0:
                elapsed = time.time() - epoch_start
                speed = (batch_idx + 1) / max(elapsed, 1e-9)
                recon_loss = nn.functional.mse_loss(x_hat, x).item()
                msg = f"[Epoch {epoch+1}/{config['epochs']}] [Batch {batch_idx}/{len(train_loader)}] Speed: {speed:.2f} it/s loss={recon_loss:.6f}"
                if is_adversarial:
                    msg += f", G_loss={g_loss.item():.6f}, D_loss={d_loss.item():.6f}"
                print(msg)

        train_mse, train_psnr, train_ssim = evaluate(train_loader, model, device, noise_std=noise_std)
        val_mse, val_psnr, val_ssim = evaluate(val_loader, model, device, noise_std=noise_std, seed=1234 + epoch)

        val_loader_preview = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=num_workers)
        save_images(model, val_loader_preview, device,
                    os.path.join(result_dir, "images", f"epoch_{epoch+1}.png"),
                    latent_noise=True, noise_std=noise_std)

        history["mse_train"].append(train_mse)
        history["mse_val"].append(val_mse)
        history["psnr_train"].append(train_psnr)
        history["psnr_val"].append(val_psnr)
        history["ssim_train"].append(train_ssim)
        history["ssim_val"].append(val_ssim)
        if is_adversarial:
            history["loss_G"].append(g_loss.item())
            history["loss_D"].append(d_loss.item())

        with open(metrics_path, "a") as f:
            f.write("\t".join(str(history[k][-1]) for k in history_keys) + "\n")

        print("[Metrics]", f"Train MSE: {train_mse:.6f}", f"Train PSNR: {train_psnr:.6f}", f"Train SSIM: {train_ssim:.6f}")
        print("[Metrics]", f"Val   MSE: {val_mse:.6f}", f"Val   PSNR: {val_psnr:.6f}", f"Val   SSIM: {val_ssim:.6f}")

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

