import os
import json
import time
import torch
import random
from torch.utils.data import DataLoader, Subset
from torchmetrics import MeanSquaredError, PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure

from data.data_setter import get_subnet_datasets, get_imagenet_datasets
from utils.helpers import get_device, save_images, plot_metrics


def train_model(model_class, config_path, input_variant="noisy-latent", dataset_variant="subset", log=False):
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
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    criterion = torch.nn.MSELoss()

    result_dir = os.path.join("results", config["name"] + "_noisy_latent", "training")
    os.makedirs(os.path.join(result_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(result_dir, "plots"), exist_ok=True)

    history = {key: [] for key in ["mse_train", "mse_val", "psnr_train", "psnr_val", "ssim_train", "ssim_val"]}
    metrics_path = os.path.join(result_dir, "metrics.txt")

    with open(metrics_path, "w") as f:
        f.write("mse_train\tmse_val\tpsnr_train\tpsnr_val\tssim_train\tssim_val\n")

    for epoch in range(config["epochs"]):
        model.train()
        epoch_start = time.time()

        for i, (x, _) in enumerate(train_loader):
            x = x.to(device)
            z = model.encode(x)
            z_noisy = z + noise_std * torch.randn_like(z)
            output = model.decode(z_noisy)

            loss = criterion(output, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if log and i % max(1, len(train_loader) // 10) == 0:
                elapsed = time.time() - epoch_start
                speed = (i + 1) / elapsed
                print(f"[Epoch {epoch+1}/{config['epochs']}] Batch {i}/{len(train_loader)} – Speed: {speed:.1f} it/s")

        # === Walidacja na podzbiorze ===
        val_indices = random.sample(range(len(val_set)), max(1, int(len(val_set) * val_fraction)))
        val_subset = Subset(val_set, val_indices)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True)

        def compute_metrics(dataloader):
            mse_metric = MeanSquaredError().to(device)
            psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
            ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

            model.eval()
            with torch.no_grad():
                for x, _ in dataloader:
                    x = x.to(device)
                    z = model.encode(x)
                    z_noisy = z + noise_std * torch.randn_like(z)
                    x_hat = model.decode(z_noisy).clamp(0, 1)

                    mse_metric.update(x_hat, x)
                    psnr_metric.update(x_hat, x)
                    ssim_metric.update(x_hat, x)

            return mse_metric.compute().item(), psnr_metric.compute().item(), ssim_metric.compute().item()

        mse_train, psnr_train, ssim_train = compute_metrics(train_loader)
        mse_val, psnr_val, ssim_val = compute_metrics(val_loader)

        for k, v in zip(history.keys(), [mse_train, mse_val, psnr_train, psnr_val, ssim_train, ssim_val]):
            history[k].append(v)

        epoch_duration = time.time() - epoch_start

        print(f"[Epoch {epoch+1}] MSE: {mse_val:.4f}, PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}")
        print(f"[Epoch {epoch+1}] Epoch time: {epoch_duration:.2f}s")

        save_images(model, val_loader, device,
                    save_path=os.path.join(result_dir, "images", f"epoch_{epoch+1}.png"),
                    num_images=4, latent_noise=True, noise_std=noise_std)

        with open(metrics_path, "a") as f:
            f.write(f"{mse_train:.5f}\t{mse_val:.5f}\t{psnr_train:.2f}\t{psnr_val:.2f}\t{ssim_train:.4f}\t{ssim_val:.4f}\n")

        plot_metrics(history, os.path.join(result_dir, "plots"))

        # Early stopping
        if mse_val + 1e-6 < best_val_mse:
            best_val_mse = mse_val
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs (no improvement in {patience} epochs).")
                break

    # === Zapis wytrenowanego modelu ===
    if "pretrained_path" in config:
        os.makedirs(os.path.dirname(config["pretrained_path"]), exist_ok=True)
        torch.save(model.state_dict(), config["pretrained_path"])
        print(f"Model weights saved to: {config['pretrained_path']}")

    print("Training with noisy latent space complete.")