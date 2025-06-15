import os
import json
import torch
from torch.utils.data import DataLoader
from math import log10
from torchmetrics import MeanSquaredError, PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure

from data.data_setter import get_subnet_datasets, get_imagenet_datasets
from utils.helpers import get_device, save_images


def evaluate_model(model_class, config_path, input_variant="clean", dataset_variant="subset", log=False):
    with open(config_path, "r") as f:
        config = json.load(f)

    device = get_device()

    if dataset_variant == "subset":
        _, val_set = get_subnet_datasets("datasets/subset_imagenet/", image_size=config["image_size"])
    else:
        _, val_set = get_imagenet_datasets("/raid/kszyc/datasets/ImageNet2012", image_size=config["image_size"])

    batch_size = 1
    num_workers = config["num_workers"]
    noise_std = config.get("noise_std", 0.1)

    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    model = model_class(config).to(device)
    model.load_state_dict(torch.load(config["pretrained_path"], map_location=device))
    model.eval()

    subname = {
        "clean": "",
        "noisy": "_noisy",
        "noisy-latent": "_noisy_latent"
    }[input_variant]

    result_dir = os.path.join("results", config["name"] + subname, "test")
    os.makedirs(os.path.join(result_dir, "images"), exist_ok=True)
    metrics_path = os.path.join(result_dir, "metrics.txt")

    # TorchMetrics
    mse_metric = MeanSquaredError().to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    with torch.no_grad(), open(metrics_path, "w") as f_out:
        f_out.write("idx\tmse\tpsnr\tssim\n")

        for idx, (x, _) in enumerate(val_loader):
            x = x.to(device)

            if input_variant == "noisy":
                x_input = torch.clamp(x + noise_std * torch.randn_like(x), 0., 1.)
                x_hat = model(x_input)
            elif input_variant == "noisy-latent":
                z = model.encode(x)
                z_noisy = z + noise_std * torch.randn_like(z)
                x_hat = model.decode(z_noisy)
            else:
                x_hat = model(x)

            x_hat = x_hat.clamp(0, 1)

            mse_metric.update(x_hat, x)
            psnr_metric.update(x_hat, x)
            ssim_metric.update(x_hat, x)

            mse = torch.nn.functional.mse_loss(x_hat, x).item()
            psnr = 10 * log10(1.0 / (mse + 1e-10))
            ssim = ssim_metric(x_hat, x).item()

            f_out.write(f"{idx}\t{mse:.5f}\t{psnr:.2f}\t{ssim:.4f}\n")

        # Final average
        mse_avg = mse_metric.compute().item()
        psnr_avg = psnr_metric.compute().item()
        ssim_avg = ssim_metric.compute().item()

        f_out.write(f"avg\t{mse_avg:.5f}\t{psnr_avg:.2f}\t{ssim_avg:.4f}\n")

    if log:
        print(f"\nEvaluation results ({input_variant}):")
        print(f"  MSE:  {mse_avg:.6f}")
        print(f"  PSNR: {psnr_avg:.2f}")
        print(f"  SSIM: {ssim_avg:.4f}")

    save_images(model, val_loader, device,
                save_path=os.path.join(result_dir, "images", "examples.png"),
                num_images=10,
                add_noise=(input_variant == "noisy"),
                latent_noise=(input_variant == "noisy-latent"),
                noise_std=noise_std)