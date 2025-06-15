import os
import json
import time
import torch
import random
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
            x_noisy = torch.clamp(x + noise_std * torch.randn_like(x), 0., 1.)
            x_hat = model(x_noisy).clamp(0, 1)

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

    suffix = f"_{input_variant}"
    result_dir = os.path.join("results", config["name"] + suffix, "training")
    os.makedirs(os.path.join(result_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(result_dir, "plots"), exist_ok=True)

    history = {key: [] for key in ["mse_train", "mse_val", "psnr_train", "psnr_val", "ssim_train", "ssim_val"]}
    metrics_path = os.path.join(result_dir, "metrics.txt")

    with open(metrics_path, "w") as f:
        f.write("mse_train\tmse_val\tpsnr_train\tpsnr_val\tssim_train\tssim_val\n")

    pretrained_path = config.get("pretrained_path", os.path.join("checkpoints", config["name"] + suffix + ".pth"))
    os.makedirs(os.path.dirname(pretrained_path), exist_ok=True)

    for epoch in range(config["epochs"]):
        model.train()
        epoch_start = time.time()

        for i, (x, _) in enumerate(train_loader):
            x = x.to(device)
            x_noisy = torch.clamp(x + noise_std * torch.randn_like(x), 0., 1.)
            output = model(x_noisy)

            loss = criterion(output, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if log and i % max(1, len(train_loader) // 10) == 0:
                elapsed = time.time() - epoch_start
                speed = (i + 1) / elapsed
                print(f"[Epoch {epoch+1}/{config['epochs']}] Batch {i}/{len(train_loader)} – Speed: {speed:.1f} it/s")

        val_size = len(val_set)
        val_subset_size = max(1, int(val_size * val_fraction))
        val_indices = random.sample(range(val_size), val_subset_size)
        val_subset = Subset(val_set, val_indices)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True)

        mse_train, psnr_train, ssim_train = evaluate(train_loader, model, device, noise_std)
        mse_val, psnr_val, ssim_val = evaluate(val_loader, model, device, noise_std)

        for k, v in zip(history.keys(), [mse_train, mse_val, psnr_train, psnr_val, ssim_train, ssim_val]):
            history[k].append(v)

        epoch_duration = time.time() - epoch_start

        print(f"[Epoch {epoch+1}] MSE: {mse_val:.4f}, PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}")
        print(f"[Epoch {epoch+1}] Epoch time: {epoch_duration:.2f}s")

        save_images(model, val_loader, device,
                    save_path=os.path.join(result_dir, "images", f"epoch_{epoch+1}.png"),
                    num_images=4, add_noise=True, noise_std=noise_std)

        with open(metrics_path, "a") as f:
            f.write(f"{mse_train:.5f}\t{mse_val:.5f}\t{psnr_train:.2f}\t{psnr_val:.2f}\t{ssim_train:.4f}\t{ssim_val:.4f}\n")

        plot_metrics(history, os.path.join(result_dir, "plots"))

        # Early stopping + save best model
        if mse_val + 1e-6 < best_val_mse:
            best_val_mse = mse_val
            epochs_no_improve = 0
            torch.save(model.state_dict(), pretrained_path)
            print(f"[Epoch {epoch+1}] New best model saved to: {pretrained_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs (no improvement in {patience} epochs).")
                break

    print("Training with noisy input complete.")