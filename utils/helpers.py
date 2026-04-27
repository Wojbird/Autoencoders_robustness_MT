from dataclasses import dataclass

import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.regression import MeanSquaredError


@dataclass
class EvalResult:
    loss: float
    mse: float
    psnr: float
    ssim: float


def make_metrics(device: torch.device):
    mse = MeanSquaredError().to(device)
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    return mse, psnr, ssim


def get_vq_reg_loss(model: torch.nn.Module) -> torch.Tensor:
    device = next(model.parameters()).device
    zero = torch.tensor(0.0, device=device)

    if hasattr(model, "vq_loss"):
        loss = getattr(model, "vq_loss")
        return loss if torch.is_tensor(loss) else zero

    if hasattr(model, "vq_losses"):
        losses = getattr(model, "vq_losses")
        if isinstance(losses, (list, tuple)):
            return sum((x for x in losses if torch.is_tensor(x)), zero)
        if torch.is_tensor(losses):
            return losses

    return zero


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_device(gpu: int | None = None) -> torch.device:
    if torch.cuda.is_available():
        if gpu is not None:
            return torch.device(f"cuda:{gpu}")
        return torch.device("cuda")
    return torch.device("cpu")