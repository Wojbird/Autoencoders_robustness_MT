import torch
import torch.nn.functional as F
from math import log10
from torchmetrics.image import StructuralSimilarityIndexMeasure


# === ŚREDNIE METRYKI ===

def calculate_mse(model, dataloader, device, add_noise=False, noise_std=0.1):
    model.eval()
    total_loss = 0
    total = 0
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            x_input = torch.clamp(x + noise_std * torch.randn_like(x), 0., 1.) if add_noise else x
            out = model(x_input)
            total_loss += F.mse_loss(out, x, reduction='sum').item()
            total += x.size(0)
    return total_loss / (total * x[0].numel())


def calculate_psnr(model, dataloader, device, add_noise=False, noise_std=0.1):
    mse = calculate_mse(model, dataloader, device, add_noise, noise_std)
    return 10 * log10(1.0 / (mse + 1e-10))


def calculate_ssim(model, dataloader, device, add_noise=False, noise_std=0.1):
    model.eval()
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    total = 0
    ssim_sum = 0
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            x_input = torch.clamp(x + noise_std * torch.randn_like(x), 0., 1.) if add_noise else x
            out = model(x_input).clamp(0, 1)
            ssim_sum += ssim_metric(out, x).item() * x.size(0)
            total += x.size(0)
    return ssim_sum / total


def calculate_mse_latent(model, dataloader, device, noise_std=0.1):
    model.eval()
    total_loss = 0
    total = 0
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            z = model.encode(x)
            z_noisy = z + noise_std * torch.randn_like(z)
            out = model.decode(z_noisy)
            total_loss += F.mse_loss(out, x, reduction='sum').item()
            total += x.size(0)
    return total_loss / (total * x[0].numel())


def calculate_psnr_latent(model, dataloader, device, noise_std=0.1):
    mse = calculate_mse_latent(model, dataloader, device, noise_std)
    return 10 * log10(1.0 / (mse + 1e-10))


def calculate_ssim_latent(model, dataloader, device, noise_std=0.1):
    model.eval()
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    total = 0
    ssim_sum = 0
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            z = model.encode(x)
            z_noisy = z + noise_std * torch.randn_like(z)
            out = model.decode(z_noisy).clamp(0, 1)
            ssim_sum += ssim_metric(out, x).item() * x.size(0)
            total += x.size(0)
    return ssim_sum / total