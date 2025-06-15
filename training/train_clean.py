import os
import json
import time
import torch
import random
from torch.utils.data import DataLoader, Subset
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.functional import peak_signal_noise_ratio
import torch.nn.functional as F

from data.data_setter import get_subnet_datasets, get_imagenet_datasets
from utils.helpers import get_device, save_images, plot_metrics


def evaluate_all_metrics(model, dataloader, device):
    model.eval()
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    total_mse = 0
    total_psnr = 0
    total_ssim = 0
    total_samples = 0

    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            out = model(x).clamp(0, 1)

            # MSE
            mse_batch = F.mse_loss(out, x, reduction='sum')
            total_mse += mse_batch.item()

            # PSNR
            psnr_batch = peak_signal_noise_ratio(out, x, data_range=1.0, reduction='sum')
            total_psnr += psnr_batch.item()

            # SSIM
            ssim_batch = ssim_metric(out, x) * x.size(0)
            total_ssim += ssim_batch.item()

            total_samples += x.size(0)

    num_pixels = x[0].numel()
    mse = total_mse / (total_samples * num_pixels)
    psnr = total_psnr / total_samples
    ssim = total_ssim / total_samples
    return mse, psnr, ssim


def train_model(model_class, config_path, input_variant="clean", dataset_variant="subset", log=False):
    with open(config_path, "r") as f:
        config = json.load(f)

    device = get_device()

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
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    criterion = torch.nn.MSELoss()

    result_dir = os.path.join("results", config["name"], "training")
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
            output = model(x)
            loss = criterion(output, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if log and i % max(1, len(train_loader) // 10) == 0:
                elapsed = time.time() - epoch_start
                speed = (i + 1) / elapsed
                print(f"[Epoch {epoch+1}/{config['epochs']}] Batch {i}/{len(train_loader)} – Speed: {speed:.1f} it/s")

        # Validation on random subset
        val_size = len(val_set)
        val_subset_size = max(1, int(val_size * val_fraction))
        val_indices = random.sample(range(val_size), val_subset_size)
        val_subset = Subset(val_set, val_indices)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True)

        mse_train, psnr_train, ssim_train = evaluate_all_metrics(model, train_loader, device)
        mse_val, psnr_val, ssim_val = evaluate_all_metrics(model, val_loader, device)

        for k, v in zip(history.keys(), [mse_train, mse_val, psnr_train, psnr_val, ssim_train, ssim_val]):
            history[k].append(v)

        epoch_duration = time.time() - epoch_start

        print(f"[Epoch {epoch+1}] MSE: {mse_val:.4f}, PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}")
        print(f"[Epoch {epoch+1}] Epoch time: {epoch_duration:.2f}s")

        save_images(model, val_loader, device,
                    save_path=os.path.join(result_dir, "images", f"epoch_{epoch+1}.png"),
                    num_images=4)

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

    print("Training with clean input complete.")