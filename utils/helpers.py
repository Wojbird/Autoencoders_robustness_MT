import os
import json
import random
import torch
import csv
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import shutil
import tempfile

from typing import Dict
from matplotlib.ticker import MaxNLocator
from dataclasses import dataclass
from torchvision.utils import make_grid
from torch.utils.data import random_split
from torchmetrics import MeanSquaredError
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(gpu_id: int | None = None):
    if torch.cuda.is_available():
        if gpu_id is None:
            return torch.device("cuda")
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def setup_device(gpu_id: int | None = None):
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True

    if torch.cuda.is_available() and gpu_id is not None:
        if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
            raise ValueError(
                f"Invalid gpu_id={gpu_id}. Available GPU ids: 0..{torch.cuda.device_count() - 1}"
            )
        torch.cuda.set_device(gpu_id)


def save_metrics(metrics_dict, save_path):
    with open(save_path, "w") as f:
        for k, v in metrics_dict.items():
            if isinstance(v, list):
                f.write(f"{k}: {','.join(f'{x:.6f}' for x in v)}\n")
            else:
                f.write(f"{k}: {v:.6f}\n")


def plot_metrics(metrics_hist: dict, results_dir: str):
    os.makedirs(results_dir, exist_ok=True)

    metric_specs = [
        ("loss", "Loss (MSE-based)", "loss_train", "loss_val"),
        ("mse", "MSE", "mse_train", "mse_val"),
        ("psnr", "PSNR", "psnr_train", "psnr_val"),
        ("ssim", "SSIM", "ssim_train", "ssim_val"),
    ]

    for fname, title, k_train, k_val in metric_specs:
        has_train = k_train in metrics_hist and len(metrics_hist[k_train]) > 0
        has_val = k_val in metrics_hist and len(metrics_hist[k_val]) > 0

        if not (has_train or has_val):
            continue

        fig = plt.figure()
        ax = fig.add_subplot(111)

        if has_train:
            ax.plot(metrics_hist[k_train], label="train")
        if has_val:
            ax.plot(metrics_hist[k_val], label="val")

        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel(fname)
        ax.legend()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        out_path = os.path.join(results_dir, f"{fname}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def _unwrap_tensor(x):
    if isinstance(x, (tuple, list)):
        return x[0]
    return x


def _forward_with_latent_noise(model, x, noise_std=0.0):
    if hasattr(model, "encode") and callable(getattr(model, "encode")) and \
       hasattr(model, "decode") and callable(getattr(model, "decode")):
        z = model.encode(x)
        z = _unwrap_tensor(z)
        z_noisy = z + noise_std * torch.randn_like(z) if noise_std > 0 else z
        out = model.decode(z_noisy)
        return _unwrap_tensor(out)

    if hasattr(model, "encoder") and hasattr(model, "decoder"):
        z = model.encoder(x)
        z = _unwrap_tensor(z)
        z_noisy = z + noise_std * torch.randn_like(z) if noise_std > 0 else z
        out = model.decoder(z_noisy)
        return _unwrap_tensor(out)

    raise AttributeError(
        f"{model.__class__.__name__} must expose either encode()/decode() "
        f"or encoder/decoder for latent-noise visualization."
    )


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
                out_tensor = _forward_with_latent_noise(model, x, noise_std)
            else:
                out = model(x)
                out_tensor = _unwrap_tensor(out)

            out_tensor = out_tensor.clamp(0., 1.).cpu()
            x_vis = x_vis.cpu()

            for i in range(min(x_vis.shape[0], num_images - images_shown)):
                images_orig.append(x_vis[i])
                images_recon.append(out_tensor[i])
                images_shown += 1

            if images_shown >= num_images:
                break

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


def ensure_val_fraction(val_set, fraction: float, split_seed: int = 42):
    if fraction >= 1.0:
        return val_set
    n = len(val_set)
    k = max(1, int(n * fraction))
    generator = torch.Generator().manual_seed(split_seed)
    subset, _ = random_split(val_set, [k, n - k], generator=generator)
    return subset


def make_results_dir(model_name: str, dataset_type: str, variant: str) -> str:
    return os.path.join("results", model_name, dataset_type, variant)


def init_csv_logger(csv_path: str, extra_columns=None):
    if extra_columns is None:
        extra_columns = []

    new_file = not os.path.exists(csv_path)
    f = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(f)

    if new_file:
        writer.writerow([
            "epoch",
            "train_loss", "train_mse", "train_psnr", "train_ssim",
            "val_loss", "val_mse", "val_psnr", "val_ssim",
            *extra_columns,
        ])
    return f, writer


@dataclass
class EvalResult:
    loss: float
    mse: float
    psnr: float
    ssim: float


def get_vq_reg_loss(model: nn.Module) -> torch.Tensor:
    device = next(model.parameters()).device

    if hasattr(model, "vq_loss") and getattr(model, "vq_loss") is not None:
        v = getattr(model, "vq_loss")
        if torch.is_tensor(v):
            return v.to(device)
        return torch.tensor(float(v), device=device)

    if hasattr(model, "vq_losses") and getattr(model, "vq_losses") is not None:
        v = getattr(model, "vq_losses")
        if isinstance(v, dict):
            vals = list(v.values())
        else:
            vals = list(v)

        if len(vals) == 0:
            return torch.zeros((), device=device)

        out = torch.zeros((), device=device)
        for t in vals:
            out = out + (t.to(device) if torch.is_tensor(t) else torch.tensor(float(t), device=device))
        return out

    return torch.zeros((), device=device)


def make_metrics(device: torch.device):
    mse = MeanSquaredError().to(device)
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    return mse, psnr, ssim


def safe_save_state_dict(state_dict, final_ckpt_path: str, retries: int = 3, delay: float = 2.0):
    final_dir = os.path.dirname(final_ckpt_path)
    os.makedirs(final_dir, exist_ok=True)

    scratch_dir = (
        os.environ.get("SLURM_TMPDIR")
        or os.environ.get("TMPDIR")
        or tempfile.gettempdir()
    )
    os.makedirs(scratch_dir, exist_ok=True)

    local_tmp_path = os.path.join(
        scratch_dir,
        os.path.basename(final_ckpt_path) + f".pid{os.getpid()}.tmp"
    )
    final_tmp_path = final_ckpt_path + ".tmp"

    last_error = None

    for attempt in range(1, retries + 1):
        try:
            torch.save(state_dict, local_tmp_path)
            shutil.copy2(local_tmp_path, final_tmp_path)
            os.replace(final_tmp_path, final_ckpt_path)

            if os.path.exists(local_tmp_path):
                os.remove(local_tmp_path)
            return

        except Exception as e:
            last_error = e

            for path in (local_tmp_path, final_tmp_path):
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

            if attempt < retries:
                time.sleep(delay)

    raise last_error