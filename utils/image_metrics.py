from dataclasses import dataclass

import torch
from torchmetrics.functional.image import (
    structural_similarity_index_measure,
)


@dataclass(frozen=True)
class PairMetrics:
    mse: torch.Tensor
    psnr: torch.Tensor
    ssim: torch.Tensor


def _validate_image_pair(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> None:
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction shape {tuple(prediction.shape)} does not match "
            f"target shape {tuple(target.shape)}."
        )

    if prediction.ndim != 4:
        raise ValueError(
            "Expected image tensors with shape [B, C, H, W], "
            f"received {tuple(prediction.shape)}."
        )

    if prediction.shape[0] == 0:
        raise ValueError("Cannot compute metrics for an empty batch.")

    if prediction.device != target.device:
        raise ValueError("Prediction and target must use the same device.")


def compute_pair_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> PairMetrics:
    """Compute MSE, PSNR and SSIM separately for every image."""
    _validate_image_pair(prediction, target)

    prediction = prediction.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)

    mse = (
        (prediction - target)
        .square()
        .flatten(start_dim=1)
        .mean(dim=1)
    )

    # data_range=1.0; exact reconstructions are capped at 120 dB.
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
                "TorchMetrics returned a batch-aggregated SSIM value "
                "despite reduction='none'. Use a compatible, pinned "
                "TorchMetrics version."
            )

        ssim = ssim.reshape(1)

    ssim = ssim.reshape(-1)

    if ssim.shape[0] != prediction.shape[0]:
        raise RuntimeError(
            f"Unexpected SSIM shape {tuple(ssim.shape)} "
            f"for batch size {prediction.shape[0]}."
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
    Compute per-image metrics for:
      1. attacked input versus clean target,
      2. reconstruction versus clean target,
      3. reconstruction versus attacked input.
    """
    _validate_image_pair(attacked_input, clean_target)
    _validate_image_pair(reconstruction, clean_target)

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

    combined = compute_pair_metrics(predictions, targets)

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