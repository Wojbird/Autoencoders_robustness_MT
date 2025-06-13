import torch
import torch.nn.functional as F
from torchvision.transforms.functional import resize
from skimage.metrics import structural_similarity as ssim_metric
import numpy as np


def _compute_ssim(img1, img2):
    # Assumes input images are torch tensors on CPU, in range [0,1], shape [C,H,W]
    img1 = img1.permute(1, 2, 0).numpy()
    img2 = img2.permute(1, 2, 0).numpy()
    return ssim_metric(img1, img2, data_range=1.0, channel_axis=2)


def calculate_mse(model, dataloader, device, add_noise=False, noise_std=0.1):
    model.eval()
    total_loss = 0
    total = 0
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            if add_noise:
                x_input = x + noise_std * torch.randn_like(x)
                x_input = torch.clamp(x_input, 0., 1.)
            else:
                x_input = x
            out = model(x_input)
            total_loss += F.mse_loss(out, x, reduction='sum').item()
            total += x.size(0)
    return total_loss / (total * x[0].numel())


def calculate_psnr(model, dataloader, device, add_noise=False, noise_std=0.1):
    mse = calculate_mse(model, dataloader, device, add_noise, noise_std)
    return 10 * torch.log10(torch.tensor(1.0) / torch.tensor(mse)).item()


def calculate_ssim(model, dataloader, device, add_noise=False, noise_std=0.1):
    model.eval()
    ssim_total = 0
    total = 0
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            if add_noise:
                x_input = x + noise_std * torch.randn_like(x)
                x_input = torch.clamp(x_input, 0., 1.)
            else:
                x_input = x
            out = model(x_input)
            for i in range(x.size(0)):
                ssim_val = _compute_ssim(x[i].cpu(), out[i].cpu().clamp(0, 1))
                ssim_total += ssim_val
                total += 1
    return ssim_total / total


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
    return 10 * torch.log10(torch.tensor(1.0) / torch.tensor(mse)).item()


def calculate_ssim_latent(model, dataloader, device, noise_std=0.1):
    model.eval()
    ssim_total = 0
    total = 0
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            z = model.encode(x)
            z_noisy = z + noise_std * torch.randn_like(z)
            out = model.decode(z_noisy)
            for i in range(x.size(0)):
                ssim_val = _compute_ssim(x[i].cpu(), out[i].cpu().clamp(0, 1))
                ssim_total += ssim_val
                total += 1
    return ssim_total / total