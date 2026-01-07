import os
import json
import random
import torch
import csv
import numpy as np
from typing import Dict
import matplotlib.pyplot as plt
from dataclasses import dataclass
from torchvision.utils import make_grid
from torch.utils.data import random_split


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_device():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True


def save_metrics(metrics_dict, save_path):
    with open(save_path, "w") as f:
        for k, v in metrics_dict.items():
            if isinstance(v, list):
                f.write(f"{k}: {','.join(f'{x:.6f}' for x in v)}\n")
            else:
                f.write(f"{k}: {v:.6f}\n")


def plot_metrics(metrics_dict, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    for metric in ["mse", "psnr", "ssim"]:
        train_key = f"{metric}_train"
        val_key = f"{metric}_val"

        if train_key in metrics_dict and val_key in metrics_dict:
            plt.figure()
            plt.plot(metrics_dict[train_key], label="Train")
            plt.plot(metrics_dict[val_key], label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel(metric.upper())
            plt.title(f"{metric.upper()} over Epochs")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, f"{metric}.png"))
            plt.close()


def save_images(model, dataloader, device, save_path,
                num_images=8, add_noise=False, latent_noise=False, noise_std=0.1):
    model.eval()
    images_shown = 0
    images_orig = []
    images_recon = []

    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            x_vis = x.clone()

            if add_noise:
                x = x + noise_std * torch.randn_like(x)
                x = torch.clamp(x, 0., 1.)

            if latent_noise:
                z = model.encode(x)
                z = z + noise_std * torch.randn_like(z)
                out = model.decode(z)
            else:
                out = model(x)

            # Obsłuż typ tuple
            out_tensor = out[0] if isinstance(out, (tuple, list)) else out
            out_tensor = out_tensor.clamp(0., 1.).cpu()
            x_vis = x_vis.cpu()

            for i in range(min(x_vis.shape[0], num_images - images_shown)):
                images_orig.append(x_vis[i])
                images_recon.append(out_tensor[i])
                images_shown += 1

            if images_shown >= num_images:
                break

    # Stwórz siatkę obrazów
    pairs = []
    for i in range(images_shown):
        pair = torch.stack([images_orig[i], images_recon[i]])
        pairs.append(pair)

    all_images = torch.cat(pairs, dim=0)
    grid = make_grid(all_images, nrow=2, padding=10)

    plt.figure(figsize=(num_images * 2, 4))
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0))
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

@dataclass
class EarlyStopping:
    patience: int
    min_delta: float = 0.0

    best: float = float("inf")
    bad_epochs: int = 0

    def step(self, value: float) -> bool:
        if value < self.best - self.min_delta:
            self.best = value
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_val_fraction(val_set, fraction: float):
    if fraction >= 1.0:
        return val_set
    n = len(val_set)
    k = max(1, int(n * fraction))
    subset, _ = random_split(val_set, [k, n - k])
    return subset


def make_results_dir(model_name: str, dataset_type: str, variant: str) -> str:
    return os.path.join("results", model_name, dataset_type, variant)


def init_csv_logger(csv_path: str):
    new_file = not os.path.exists(csv_path)
    f = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(f)
    if new_file:
        writer.writerow([
            "epoch",
            "train_loss", "train_mse", "train_psnr", "train_ssim",
            "val_loss", "val_mse", "val_psnr", "val_ssim",
        ])
    return f, writer