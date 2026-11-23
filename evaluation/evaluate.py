import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.helpers import EvalResult, get_vq_reg_loss
from utils.image_metrics import compute_pair_metrics
from utils.latent import (
    add_gaussian_latent_noise,
    forward_with_latent_transform,
    unwrap_tensor,
)


def _randn_like_seeded(
    x: torch.Tensor,
    seed: Optional[int],
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
    noise_std: float,
    noise_seed: Optional[int] = None,
    batch_idx: int = 0,
    relative: bool = True,
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


@torch.no_grad()
def evaluate_reconstruction(
    model: nn.Module,
    dataloader,
    device: torch.device,
    *,
    variant: str,
    noise_std: float = 0.0,
    latent_noise: bool = False,
    max_batches: Optional[int] = None,
    noise_seed: Optional[int] = None,
    latent_noise_relative: bool = True,
) -> EvalResult:
    if variant not in {"clean", "noisy", "noisy_latent"}:
        raise ValueError(
            "variant must be one of: 'clean', 'noisy', 'noisy_latent'."
        )

    if noise_std < 0.0:
        raise ValueError("noise_std must be non-negative.")

    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive or None.")

    use_latent_noise = latent_noise or variant == "noisy_latent"

    if use_latent_noise and variant == "noisy":
        raise ValueError(
            "variant='noisy' cannot be combined with latent_noise=True."
        )

    was_training = model.training
    model.eval()

    mse_sum = 0.0
    psnr_sum = 0.0
    ssim_sum = 0.0
    number_of_samples = 0

    try:
        for batch_idx, (x, _) in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            x = x.to(device)

            if use_latent_noise:
                x_hat = _forward_with_latent_noise(
                    model,
                    x,
                    noise_std=noise_std,
                    noise_seed=noise_seed,
                    batch_idx=batch_idx,
                    relative=latent_noise_relative,
                )
            else:
                if variant == "noisy" and noise_std > 0.0:
                    noise = _randn_like_seeded(
                        x,
                        seed=noise_seed,
                        offset=batch_idx,
                    )
                    model_input = torch.clamp(
                        x + noise_std * noise,
                        0.0,
                        1.0,
                    )
                else:
                    model_input = x

                x_hat = unwrap_tensor(model(model_input))

            x_hat = torch.clamp(x_hat, 0.0, 1.0)

            batch_metrics = compute_pair_metrics(
                prediction=x_hat,
                target=x,
            )

            mse_sum += float(batch_metrics.mse.sum().item())
            psnr_sum += float(batch_metrics.psnr.sum().item())
            ssim_sum += float(batch_metrics.ssim.sum().item())
            number_of_samples += x.shape[0]

    finally:
        model.train(was_training)

    if number_of_samples == 0:
        raise ValueError(
            "Empty dataloader provided to evaluate_reconstruction()."
        )

    mean_mse = mse_sum / number_of_samples

    return EvalResult(
        loss=mean_mse,
        mse=mean_mse,
        psnr=psnr_sum / number_of_samples,
        ssim=ssim_sum / number_of_samples,
    )


def compute_training_loss(
    model: nn.Module,
    x_hat: torch.Tensor,
    x_target: torch.Tensor,
    *,
    allow_vq: bool,
) -> torch.Tensor:
    reconstruction_loss = F.mse_loss(x_hat, x_target)

    if not allow_vq:
        return reconstruction_loss

    return reconstruction_loss + get_vq_reg_loss(model)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader,
    device: torch.device,
    *,
    variant: str,
    noise_std: float,
    results_dir: Optional[str] = None,
    noise_seed: Optional[int] = None,
    latent_noise_relative: bool = True,
) -> dict[str, float]:
    result = evaluate_reconstruction(
        model,
        dataloader,
        device,
        variant=variant,
        noise_std=noise_std,
        latent_noise=variant == "noisy_latent",
        noise_seed=noise_seed,
        latent_noise_relative=latent_noise_relative,
    )

    output = {
        "loss": result.loss,
        "mse": result.mse,
        "psnr": result.psnr,
        "ssim": result.ssim,
    }

    if results_dir is not None:
        os.makedirs(results_dir, exist_ok=True)
        summary_path = os.path.join(
            results_dir,
            "evaluation_summary.txt",
        )

        with open(summary_path, "w", encoding="utf-8") as summary_file:
            for key, value in output.items():
                summary_file.write(f"{key}: {value:.6f}\n")

    return output