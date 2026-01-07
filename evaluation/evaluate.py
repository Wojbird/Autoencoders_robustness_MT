import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torchmetrics import MeanSquaredError
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


@dataclass
class EvalResult:
    loss: float
    mse: float
    psnr: float
    ssim: float


def _get_vq_reg_loss(model: nn.Module) -> torch.Tensor:
    """
    Returns VQ regularization loss stored inside the model (if any).
    Supports:
      - model.vq_loss (scalar tensor)
      - model.vq_losses (list/tuple of scalar tensors or dict of scalar tensors)
    If nothing exists, returns 0 on the correct device.
    """
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


def _make_metrics(device: torch.device):
    mse = MeanSquaredError().to(device)
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    return mse, psnr, ssim


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
) -> EvalResult:
    """
    Computes reconstruction loss (MSE) and image metrics (MSE/PSNR/SSIM).

    variant:
      - "clean": input x
      - "noisy": input clamp(x + noise, 0, 1)
      - "noisy_latent": latent path: encode -> add noise -> decode
    latent_noise=True forces the encode/decode path (used for noisy_latent).
    """
    model.eval()
    loss_fn = nn.MSELoss()

    mse_metric, psnr_metric, ssim_metric = _make_metrics(device)

    loss_sum = 0.0
    n_batches = 0

    for i, (x, _) in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break

        x = x.to(device)

        if latent_noise:
            if not (hasattr(model, "encode") and hasattr(model, "decode")):
                raise AttributeError("Model must implement encode() and decode() for latent noise evaluation.")
            z = model.encode(x)
            z_noisy = z + (noise_std * torch.randn_like(z) if noise_std > 0 else 0.0)
            x_hat = model.decode(z_noisy)
        else:
            if variant == "noisy" and noise_std > 0:
                x_in = torch.clamp(x + noise_std * torch.randn_like(x), 0.0, 1.0)
            else:
                x_in = x
            x_hat = model(x_in)

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
    """
    Reconstruction MSE + optional VQ regularization (if present in model).
    """
    recon = nn.MSELoss()(x_hat, x_target)
    if not allow_vq:
        return recon
    return recon + _get_vq_reg_loss(model)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader,
    device: torch.device,
    *,
    variant: str,
    noise_std: float,
    results_dir: Optional[str] = None,
) -> Dict[str, float]:
    """
    Public evaluation entry (for main.py test mode).
    Returns a flat dict with metrics.
    """
    latent = (variant == "noisy_latent")
    res = evaluate_reconstruction(
        model,
        dataloader,
        device,
        variant=variant,
        noise_std=noise_std,
        latent_noise=latent,
    )

    out = {
        "loss": res.loss,
        "mse": res.mse,
        "psnr": res.psnr,
        "ssim": res.ssim,
    }

    # Optional: save a plain text summary
    if results_dir is not None:
        os.makedirs(results_dir, exist_ok=True)
        summary_path = os.path.join(results_dir, "evaluation_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            for k, v in out.items():
                f.write(f"{k}: {v:.6f}\n")

    return out