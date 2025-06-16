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
from utils.helpers import get_device, save_images, plot_metrics


def evaluate(loader, model, device, noise_std):
    mse_metric = MeanSquaredError().to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            z = model.encode(x)
            z_noisy = z + noise_std * torch.randn_like(z)
            x_hat = model.decode(z_noisy).clamp(0, 1)

            mse_metric.update(x_hat, x)
            psnr_metric.update(x_hat, x)
            ssim_metric.update(x_hat, x)

    return (
        mse_metric.compute().item(),
        psnr_metric.compute().item(),
        ssim_metric.compute().item()
    )


def train_adversarial_unet_ae(model_class, discriminator_class, config_path, input_variant="noisy-latent", dataset_variant="subset", log=False):
    with open(config_path, "r") as f:
        config = json.load(f)

    device = get_device()

    if dataset_variant == "subset":
        train_set, val_set = get_subnet_datasets("datasets/subset_imagenet/", image_size=config["image_size"])
    else:
        train_set, val_set = get_imagenet_datasets("datasets/full_imagenet/", image_size=config["image_size"])

    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    noise_std = config.get("noise_std", 0.1)
    val_fraction = config.get("val_subset_fraction", 0.1)
    patience = config.get("early_stopping_patience", 5)
    best_val_mse = float("inf")
    epochs_no_improve = 0

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)

    model = model_class(config).to(device)
    discriminator = discriminator_class(latent_dim=config["latent_dim"]).to(device)

    optimizer_G = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    criterion_recon = nn.MSELoss()
    criterion_adv = nn.BCELoss()

    suffix = f"_{input_variant}"
    result_dir = os.path.join("results", config["name"] + suffix, "training")
    os.makedirs(os.path.join(result_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(result_dir, "plots"), exist_ok=True)

    history = {key: [] for key in ["mse_train", "mse_val", "psnr_train", "psnr_val", "ssim_train", "ssim_val",
                                  "loss_G", "loss_D"]}
    metrics_path = os.path.join(result_dir, "metrics.txt")

    with open(metrics_path, "w") as f:
        f.write("mse_train\tmse_val\tpsnr_train\tpsnr_val\tssim_train\tssim_val\tloss_G\tloss_D\n")

    pretrained_path_G = os.path.join("checkpoints", config["name"] + suffix + "_G.pth")
    pretrained_path_D = os.path.join("checkpoints", config["name"] + suffix + "_D.pth")
    os.makedirs(os.path.dirname(pretrained_path_G), exist_ok=True)
    os.makedirs(os.path.dirname(pretrained_path_D), exist_ok=True)

    for epoch in range(config["epochs"]):
        model.train()
        discriminator.train()
        epoch_start = time.time()

        for i, (x, _) in enumerate(train_loader):
            x = x.to(device)
            batch_size_curr = x.size(0)

            # === Train Discriminator ===
            optimizer_D.zero_grad()

            # Real latent samples from prior (Gaussian)
            real_z = torch.randn(batch_size_curr, config["latent_dim"], x.size(2)//32, x.size(3)//32, device=device)

            # Fake latent samples from encoder
            fake_z = model.encode(x).detach()

            real_labels = torch.ones(batch_size_curr, 1, device=device)
            fake_labels = torch.zeros(batch_size_curr, 1, device=device)

            pred_real = discriminator(real_z)
            loss_D_real = criterion_adv(pred_real, real_labels)

            pred_fake = discriminator(fake_z)
            loss_D_fake = criterion_adv(pred_fake, fake_labels)

            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()

            # === Train Generator (Autoencoder) ===
            optimizer_G.zero_grad()

            z = model.encode(x)
            x_recon = model.decode(z)

            # Reconstruction loss
            loss_recon = criterion_recon(x_recon, x)

            # Adversarial loss (try to fool discriminator)
            pred = discriminator(z)
            loss_adv = criterion_adv(pred, real_labels)

            loss_G = loss_recon + 1e-3 * loss_adv  # Weight adversarial loss lightly

            loss_G.backward()
            optimizer_G.step()

            if log and i % max(1, len(train_loader) // 10) == 0:
                elapsed = time.time() - epoch_start
                speed = (i + 1) / elapsed
                print(f"[Epoch {epoch+1}/{config['epochs']}] Batch {i}/{len(train_loader)} – Speed: {speed:.1f} it/s")
                print(f"loss_D: {loss_D.item():.4f}, loss_G: {loss_G.item():.4f}")

        # Validation
        val_indices = random.sample(range(len(val_set)), max(1, int(len(val_set) * val_fraction)))
        val_subset = Subset(val_set, val_indices)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True)

        mse_train, psnr_train, ssim_train = evaluate(train_loader, model, device, noise_std)
        mse_val, psnr_val, ssim_val = evaluate(val_loader, model, device, noise_std)

        # Save training losses for plotting
        history["mse_train"].append(mse_train)
        history["mse_val"].append(mse_val)
        history["psnr_train"].append(psnr_train)
        history["psnr_val"].append(psnr_val)
        history["ssim_train"].append(ssim_train)
        history["ssim_val"].append(ssim_val)
        history["loss_G"].append(loss_G.item())
        history["loss_D"].append(loss_D.item())

        epoch_duration = time.time() - epoch_start

        print(f"[Epoch {epoch+1}] MSE: {mse_val:.4f}, PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}")
        print(f"[Epoch {epoch+1}] loss_G: {loss_G.item():.4f}, loss_D: {loss_D.item():.4f}")
        print(f"[Epoch {epoch+1}] Epoch time: {epoch_duration:.2f}s")

        save_images(model, val_loader, device,
                    save_path=os.path.join(result_dir, "images", f"epoch_{epoch+1}.png"),
                    num_images=4, latent_noise=True, noise_std=noise_std)

        with open(metrics_path, "a") as f:
            f.write(f"{mse_train:.5f}\t{mse_val:.5f}\t{psnr_train:.2f}\t{psnr_val:.2f}\t{ssim_train:.4f}\t{ssim_val:.4f}\t{loss_G:.5f}\t{loss_D:.5f}\n")

        plot_metrics(history, os.path.join(result_dir, "plots"))

        # Early stopping + best model saving
        if mse_val + 1e-6 < best_val_mse:
            best_val_mse = mse_val
            epochs_no_improve = 0
            torch.save(model.state_dict(), pretrained_path_G)
            torch.save(discriminator.state_dict(), pretrained_path_D)
            print(f"[Epoch {epoch+1}] New best models saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs (no improvement in {patience} epochs).")
                break

    print("Adversarial UNet AE training complete.")