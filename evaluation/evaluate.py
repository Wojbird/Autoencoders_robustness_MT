import os
from typing import Dict, Optional

import torch
import torch.nn as nn
from utils.helpers import get_vq_reg_loss, make_metrics, EvalResult


def _unwrap_tensor(x):
    if isinstance(x, (tuple, list)):
        return x[0]
    return x


def _randn_like_seeded(x: torch.Tensor, seed: Optional[int], offset: int) -> torch.Tensor:
    if seed is None:
        return torch.randn_like(x)

    devices = [x.device] if x.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed + offset)
        if x.device.type == "cuda":
            torch.cuda.manual_seed_all(seed + offset)
        return torch.randn_like(x)


def _forward_with_latent_noise(
    model: nn.Module,
    x: torch.Tensor,
    noise_std: float,
    noise_seed: Optional[int] = None,
    batch_idx: int = 0,
) -> torch.Tensor:
    if hasattr(model, "encode") and callable(getattr(model, "encode")) and \
       hasattr(model, "decode") and callable(getattr(model, "decode")):
        z = model.encode(x)
        z = _unwrap_tensor(z)
        noise = _randn_like_seeded(z, noise_seed, batch_idx)
        z_noisy = z + noise_std * noise if noise_std > 0 else z
        x_hat = model.decode(z_noisy)
        return _unwrap_tensor(x_hat)

    if hasattr(model, "encoder") and hasattr(model, "decoder"):
        z = model.encoder(x)
        z = _unwrap_tensor(z)
        noise = _randn_like_seeded(z, noise_seed, batch_idx)
        z_noisy = z + noise_std * noise if noise_std > 0 else z
        x_hat = model.decoder(z_noisy)
        return _unwrap_tensor(x_hat)

    raise AttributeError(
        f"{model.__class__.__name__} must expose either encode()/decode() "
        f"or encoder/decoder for latent-noise evaluation."
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
) -> EvalResult:
    if variant not in {"clean", "noisy", "noisy_latent"}:
        raise ValueError("variant must be one of: 'clean', 'noisy', 'noisy_latent'.")

    model.eval()
    loss_fn = nn.MSELoss()
    mse_metric, psnr_metric, ssim_metric = make_metrics(device)

    loss_sum = 0.0
    n_batches = 0

    for i, (x, _) in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break

        x = x.to(device)

        if latent_noise or variant == "noisy_latent":
            x_hat = _forward_with_latent_noise(
                model,
                x,
                noise_std=noise_std,
                noise_seed=noise_seed,
                batch_idx=i,
            )
        else:
            if variant == "noisy" and noise_std > 0:
                noise = _randn_like_seeded(x, noise_seed, i)
                x_in = torch.clamp(x + noise_std * noise, 0.0, 1.0)
            else:
                x_in = x

            x_hat = model(x_in)
            x_hat = _unwrap_tensor(x_hat)

        x_hat = torch.clamp(x_hat, 0.0, 1.0)

        loss = loss_fn(x_hat, x)
        loss_sum += float(loss.item())
        n_batches += 1

        mse_metric.update(x_hat, x)
        psnr_metric.update(x_hat, x)
        ssim_metric.update(x_hat, x)

    if n_batches == 0:
        raise ValueError("Empty dataloader provided to evaluate_reconstruction().")

    return EvalResult(
        loss=loss_sum / n_batches,
        mse=float(mse_metric.compute().item()),
        psnr=float(psnr_metric.compute().item()),
        ssim=float(ssim_metric.compute().item()),
    )


def compute_training_loss(
    model: nn.Module,
    x_hat: torch.Tensor,
    x_target: torch.Tensor,
    *,
    allow_vq: bool,
) -> torch.Tensor:
    recon = nn.MSELoss()(x_hat, x_target)
    if not allow_vq:
        return recon
    return recon + get_vq_reg_loss(model)


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
) -> Dict[str, float]:
    latent = variant == "noisy_latent"
    res = evaluate_reconstruction(
        model,
        dataloader,
        device,
        variant=variant,
        noise_std=noise_std,
        latent_noise=latent,
        noise_seed=noise_seed,
    )

    out = {
        "loss": res.loss,
        "mse": res.mse,
        "psnr": res.psnr,
        "ssim": res.ssim,
    }

    if results_dir is not None:
        os.makedirs(results_dir, exist_ok=True)
        summary_path = os.path.join(results_dir, "evaluation_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            for k, v in out.items():
                f.write(f"{k}: {v:.6f}\n")

    return out