import torch
from torchmetrics import MeanSquaredError
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


def _compute_metrics(model, dataloader, device, noise_std=0.0, latent=False):
    """ Unified TorchMetrics-based evaluation for AE or latent-AE. """
    if len(dataloader) == 0:
        raise ValueError("Provided dataloader is empty. Cannot compute metrics.")

    model.eval()
    mse_metric = MeanSquaredError().to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)

            if latent:
                z = model.encode(x)
                z_noisy = z + noise_std * torch.randn_like(z)
                x_hat = model.decode(z_noisy)
            else:
                x_input = torch.clamp(x + noise_std * torch.randn_like(x), 0., 1.) if noise_std > 0 else x
                x_hat = model(x_input)

            x_hat = x_hat.clamp(0, 1)

            mse_metric.update(x_hat, x)
            psnr_metric.update(x_hat, x)
            ssim_metric.update(x_hat, x)

    mse = mse_metric.compute().item()
    psnr = psnr_metric.compute().item()
    ssim = ssim_metric.compute().item()

    return mse, psnr, ssim


def calculate_mse(model, dataloader, device, add_noise=False, noise_std=0.1):
    mse, _, _ = _compute_metrics(model, dataloader, device,
                                 noise_std if add_noise else 0.0, latent=False)
    return mse


def calculate_psnr(model, dataloader, device, add_noise=False, noise_std=0.1):
    _, psnr, _ = _compute_metrics(model, dataloader, device,
                                  noise_std if add_noise else 0.0, latent=False)
    return psnr


def calculate_ssim(model, dataloader, device, add_noise=False, noise_std=0.1):
    _, _, ssim = _compute_metrics(model, dataloader, device,
                                  noise_std if add_noise else 0.0, latent=False)
    return ssim


def calculate_mse_latent(model, dataloader, device, noise_std=0.1):
    mse, _, _ = _compute_metrics(model, dataloader, device, noise_std, latent=True)
    return mse


def calculate_psnr_latent(model, dataloader, device, noise_std=0.1):
    _, psnr, _ = _compute_metrics(model, dataloader, device, noise_std, latent=True)
    return psnr


def calculate_ssim_latent(model, dataloader, device, noise_std=0.1):
    _, _, ssim = _compute_metrics(model, dataloader, device, noise_std, latent=True)
    return ssim