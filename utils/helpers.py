import csv
import json
import os
import random
import shutil
import tempfile
import time
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.ticker import MaxNLocator
from torch.utils.data import random_split
from torchvision.utils import make_grid

from utils.latent import (
    add_gaussian_latent_noise,
    forward_with_latent_transform,
    unwrap_tensor,
)


def _ensure_parent_directory(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_gpu_id(gpu_id: int | None) -> None:
    if gpu_id is None or not torch.cuda.is_available():
        return

    device_count = torch.cuda.device_count()
    if gpu_id < 0 or gpu_id >= device_count:
        raise ValueError(
            f"Invalid gpu_id={gpu_id}. "
            f"Available GPU ids: 0..{device_count - 1}."
        )


def get_device(gpu_id: int | None = None) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")

    _validate_gpu_id(gpu_id)

    if gpu_id is None:
        return torch.device("cuda")

    return torch.device(f"cuda:{gpu_id}")


def setup_device(
    gpu_id: int | None = None,
    *,
    deterministic: bool = False,
) -> torch.device:
    torch.set_num_threads(1)

    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(
        deterministic,
        warn_only=True,
    )

    device = get_device(gpu_id)

    if device.type == "cuda" and gpu_id is not None:
        torch.cuda.set_device(gpu_id)

    return device


def save_metrics(metrics_dict: dict, save_path: str) -> None:
    _ensure_parent_directory(save_path)

    with open(save_path, "w", encoding="utf-8") as output_file:
        for key, value in metrics_dict.items():
            if isinstance(value, list):
                serialized = ",".join(
                    f"{float(item):.6f}"
                    for item in value
                )
                output_file.write(f"{key}: {serialized}\n")
            else:
                output_file.write(f"{key}: {float(value):.6f}\n")


def plot_metrics(metrics_hist: dict, results_dir: str) -> None:
    os.makedirs(results_dir, exist_ok=True)

    metric_specs = [
        ("loss", "Loss (MSE-based)", "loss_train", "loss_val"),
        ("mse", "MSE", "mse_train", "mse_val"),
        ("psnr", "PSNR", "psnr_train", "psnr_val"),
        ("ssim", "SSIM", "ssim_train", "ssim_val"),
    ]

    for filename, title, train_key, val_key in metric_specs:
        has_train = bool(metrics_hist.get(train_key))
        has_val = bool(metrics_hist.get(val_key))

        if not (has_train or has_val):
            continue

        figure = plt.figure()
        axis = figure.add_subplot(111)

        if has_train:
            axis.plot(metrics_hist[train_key], label="train")
        if has_val:
            axis.plot(metrics_hist[val_key], label="val")

        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.set_ylabel(filename)
        axis.legend()
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))

        output_path = os.path.join(
            results_dir,
            f"{filename}.png",
        )
        figure.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(figure)


def _randn_like_seeded(
    x: torch.Tensor,
    seed: int | None,
    offset: int,
) -> torch.Tensor:
    if seed is None:
        return torch.randn_like(x)

    generator = torch.Generator(device=x.device)
    generator.manual_seed(int(seed) + int(offset))

    return torch.randn(
        x.shape,
        dtype=x.dtype,
        device=x.device,
        generator=generator,
    )


def _forward_with_latent_noise(
    model: nn.Module,
    x: torch.Tensor,
    noise_std: float = 0.0,
    relative: bool = True,
    noise_seed: int | None = None,
    batch_idx: int = 0,
) -> torch.Tensor:
    def transform(z: torch.Tensor) -> torch.Tensor:
        noise = _randn_like_seeded(
            z,
            seed=noise_seed,
            offset=batch_idx,
        )

        return add_gaussian_latent_noise(
            z,
            noise_std=noise_std,
            relative=relative,
            noise=noise,
        )

    return forward_with_latent_transform(
        model,
        x,
        transform,
    )


def save_images(
    model: nn.Module,
    dataloader,
    device: torch.device,
    save_path: str,
    num_images: int = 8,
    add_noise: bool = False,
    latent_noise: bool = False,
    noise_std: float = 0.1,
    latent_noise_relative: bool = True,
    noise_seed: int | None = None,
) -> None:
    if num_images <= 0:
        raise ValueError("num_images must be positive.")
    if noise_std < 0.0:
        raise ValueError("noise_std must be non-negative.")

    was_training = model.training
    model.eval()
    _ensure_parent_directory(save_path)

    images_original = []
    images_reconstructed = []

    try:
        with torch.no_grad():
            for batch_idx, (x, _) in enumerate(dataloader):
                x = x.to(device)
                x_original = x.clone()

                if add_noise:
                    input_noise = _randn_like_seeded(
                        x,
                        seed=noise_seed,
                        offset=batch_idx,
                    )
                    model_input = torch.clamp(
                        x + noise_std * input_noise,
                        0.0,
                        1.0,
                    )
                else:
                    model_input = x

                if latent_noise:
                    reconstruction = _forward_with_latent_noise(
                        model,
                        model_input,
                        noise_std=noise_std,
                        relative=latent_noise_relative,
                        noise_seed=noise_seed,
                        batch_idx=batch_idx,
                    )
                else:
                    reconstruction = unwrap_tensor(
                        model(model_input)
                    )

                reconstruction = reconstruction.clamp(0.0, 1.0).cpu()
                x_original = x_original.cpu()

                remaining = num_images - len(images_original)
                number_to_add = min(x_original.shape[0], remaining)

                for image_idx in range(number_to_add):
                    images_original.append(x_original[image_idx])
                    images_reconstructed.append(reconstruction[image_idx])

                if len(images_original) >= num_images:
                    break

    finally:
        model.train(was_training)

    if not images_original:
        raise ValueError("No images available for save_images().")

    pairs = [
        torch.stack([original, reconstruction])
        for original, reconstruction in zip(
            images_original,
            images_reconstructed,
        )
    ]

    image_grid = make_grid(
        torch.cat(pairs, dim=0),
        nrow=2,
        padding=10,
    )

    figure = plt.figure(
        figsize=(len(images_original) * 2, 4)
    )
    axis = figure.add_subplot(111)
    axis.axis("off")
    axis.imshow(image_grid.permute(1, 2, 0).numpy())
    figure.savefig(save_path, bbox_inches="tight")
    plt.close(figure)


@dataclass
class EarlyStopping:
    patience: int
    min_delta: float = 0.0
    best: float = float("inf")
    bad_epochs: int = 0

    def __post_init__(self) -> None:
        if self.patience <= 0:
            raise ValueError("patience must be positive.")
        if self.min_delta < 0.0:
            raise ValueError("min_delta must be non-negative.")

    def step(self, value: float) -> bool:
        if value < self.best - self.min_delta:
            self.best = value
            self.bad_epochs = 0
            return False

        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    if not isinstance(config, dict):
        raise ValueError("The configuration root must be a JSON object.")

    return config


def ensure_val_fraction(
    dataset,
    fraction: float,
    split_seed: int = 42,
):
    fraction = float(fraction)

    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must satisfy 0.0 < fraction <= 1.0.")

    if fraction == 1.0:
        return dataset

    dataset_size = len(dataset)
    if dataset_size == 0:
        raise ValueError("Cannot select a fraction of an empty dataset.")

    subset_size = max(1, int(dataset_size * fraction))
    generator = torch.Generator().manual_seed(split_seed)

    subset, _ = random_split(
        dataset,
        [subset_size, dataset_size - subset_size],
        generator=generator,
    )

    return subset


def make_results_dir(
    model_name: str,
    dataset_type: str,
    variant: str,
) -> str:
    return os.path.join(
        "results",
        model_name,
        dataset_type,
        variant,
    )


def init_csv_logger(
    csv_path: str,
    extra_columns=None,
    overwrite: bool = True,
):
    if extra_columns is None:
        extra_columns = []

    _ensure_parent_directory(csv_path)

    mode = "w" if overwrite else "a"
    new_file = overwrite or not os.path.exists(csv_path)

    csv_file = open(
        csv_path,
        mode,
        newline="",
        encoding="utf-8",
    )
    writer = csv.writer(csv_file)

    if new_file:
        writer.writerow([
            "epoch",
            "train_loss",
            "train_mse",
            "train_psnr",
            "train_ssim",
            "val_loss",
            "val_mse",
            "val_psnr",
            "val_ssim",
            *extra_columns,
        ])

    return csv_file, writer


@dataclass(frozen=True)
class EvalResult:
    loss: float
    mse: float
    psnr: float
    ssim: float


def get_vq_reg_loss(model: nn.Module) -> torch.Tensor:
    try:
        reference_parameter = next(model.parameters())
    except StopIteration as exc:
        raise ValueError(
            "Cannot determine the device of a model without parameters."
        ) from exc

    device = reference_parameter.device
    dtype = reference_parameter.dtype

    vq_loss = getattr(model, "vq_loss", None)
    if vq_loss is not None:
        if torch.is_tensor(vq_loss):
            return vq_loss.to(device=device)
        return torch.tensor(
            float(vq_loss),
            device=device,
            dtype=dtype,
        )

    vq_losses = getattr(model, "vq_losses", None)
    if vq_losses is not None:
        if isinstance(vq_losses, dict):
            values = list(vq_losses.values())
        elif isinstance(vq_losses, (list, tuple)):
            values = list(vq_losses)
        else:
            values = [vq_losses]

        total = torch.zeros(
            (),
            device=device,
            dtype=dtype,
        )

        for value in values:
            if torch.is_tensor(value):
                total = total + value.to(device=device)
            else:
                total = total + torch.tensor(
                    float(value),
                    device=device,
                    dtype=dtype,
                )

        return total

    return torch.zeros(
        (),
        device=device,
        dtype=dtype,
    )


def safe_save_state_dict(
    state_dict,
    final_checkpoint_path: str,
    retries: int = 3,
    delay: float = 2.0,
) -> None:
    if retries <= 0:
        raise ValueError("retries must be positive.")
    if delay < 0.0:
        raise ValueError("delay must be non-negative.")

    _ensure_parent_directory(final_checkpoint_path)

    scratch_dir = (
        os.environ.get("SLURM_TMPDIR")
        or os.environ.get("TMPDIR")
        or tempfile.gettempdir()
    )
    os.makedirs(scratch_dir, exist_ok=True)

    basename = os.path.basename(final_checkpoint_path)
    process_suffix = f".pid{os.getpid()}.tmp"
    local_temporary_path = os.path.join(
        scratch_dir,
        basename + process_suffix,
    )
    final_temporary_path = final_checkpoint_path + process_suffix

    last_error = None

    for attempt in range(1, retries + 1):
        try:
            torch.save(state_dict, local_temporary_path)
            shutil.copy2(
                local_temporary_path,
                final_temporary_path,
            )
            os.replace(
                final_temporary_path,
                final_checkpoint_path,
            )

            if os.path.exists(local_temporary_path):
                os.remove(local_temporary_path)

            return

        except Exception as error:
            last_error = error

            for path in (
                local_temporary_path,
                final_temporary_path,
            ):
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

            if attempt < retries:
                time.sleep(delay)

    if last_error is None:
        raise RuntimeError("Checkpoint saving failed for an unknown reason.")

    raise last_error