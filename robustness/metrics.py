from __future__ import annotations

from dataclasses import dataclass

import torch
from torchmetrics.functional.image import (
    structural_similarity_index_measure,
)


@dataclass
class PairMetrics:
    mse: torch.Tensor
    psnr: torch.Tensor
    ssim: torch.Tensor


def compute_pair_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> PairMetrics:
    """
    Return MSE, PSNR, and SSIM separately for every image in a batch.
    """
    prediction = torch.clamp(
        prediction,
        min=0.0,
        max=1.0,
    )
    target = torch.clamp(
        target,
        min=0.0,
        max=1.0,
    )

    mse = (
        (prediction - target)
        .square()
        .flatten(start_dim=1)
        .mean(dim=1)
    )

    psnr = 10.0 * torch.log10(
        1.0 / mse.clamp_min(1e-12)
    )

    ssim = structural_similarity_index_measure(
        prediction,
        target,
        data_range=1.0,
        reduction="none",
    )

    if isinstance(ssim, tuple):
        ssim = ssim[0]

    if ssim.ndim == 0:
        if prediction.shape[0] != 1:
            raise RuntimeError(
                "TorchMetrics returned one batch-aggregated SSIM value "
                "despite reduction='none'. Upgrade torchmetrics before "
                "running this experiment."
            )

        ssim = ssim.reshape(1)

    ssim = ssim.reshape(-1)

    if ssim.shape[0] != prediction.shape[0]:
        raise RuntimeError(
            "Unexpected SSIM output shape "
            f"{tuple(ssim.shape)} for batch size "
            f"{prediction.shape[0]}."
        )

    return PairMetrics(
        mse=mse,
        psnr=psnr,
        ssim=ssim,
    )


def compute_evaluation_metrics(
    attacked_input: torch.Tensor,
    reconstruction: torch.Tensor,
    clean_target: torch.Tensor,
) -> tuple[PairMetrics, PairMetrics, PairMetrics]:
    """
    Compute:

    1. attacked input vs clean image,
    2. reconstruction vs clean image,
    3. reconstruction vs attacked input.
    """
    batch_size = clean_target.shape[0]

    predictions = torch.cat(
        [
            attacked_input,
            reconstruction,
            reconstruction,
        ],
        dim=0,
    )

    targets = torch.cat(
        [
            clean_target,
            clean_target,
            attacked_input,
        ],
        dim=0,
    )

    combined = compute_pair_metrics(
        predictions,
        targets,
    )

    def part(start: int) -> PairMetrics:
        stop = start + batch_size

        return PairMetrics(
            mse=combined.mse[start:stop],
            psnr=combined.psnr[start:stop],
            ssim=combined.ssim[start:stop],
        )

    return (
        part(0),
        part(batch_size),
        part(2 * batch_size),
    )