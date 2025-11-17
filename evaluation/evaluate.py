import os
import json
import torch
from torch.utils.data import DataLoader
from torchmetrics import MeanSquaredError
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from data.data_setter import get_subnet_datasets, get_imagenet_datasets
from utils.helpers import get_device, save_images


def evaluate_model(model_class, config_path, input_variant="clean", dataset_variant="subset", log=False):
    with open(config_path, "r") as f:
        config = json.load(f)

    suffix = f"_{input_variant}"
    default_ckpt = os.path.join("checkpoints", config["name"] + suffix + ".pth")
    pretrained_path = config.get("pretrained_path", default_ckpt)

    device = get_device()
    model = model_class(config).to(device)

    # Wczesne pominięcie: model nieobsługiwany w noisy_latent
    if input_variant == "noisy_latent":
        if hasattr(model, "quantizer") or hasattr(model, "top_quantizer") or hasattr(model, "bottom_quantizer") \
           or hasattr(model, "vq_loss") or hasattr(model, "vq_losses") \
           or ("vq" in model.__class__.__name__.lower()):
            print(f"[INFO] Skipping evaluation: Model {model.__class__.__name__} is not supported in noisy_latent mode.")
            return

    if not os.path.exists(pretrained_path):
        print(f"[INFO] Skipping evaluation: Checkpoint not found: {pretrained_path}")
        return

    if dataset_variant == "subset":
        _, val_set = get_subnet_datasets(image_size=config["image_size"])  # Subnet of ImageNet
    else:
        _, val_set = get_imagenet_datasets(image_size=config["image_size"])  # ImageNet

    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    noise_std = config.get("noise_std", 0.1)

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Wczytaj wytrenowane wagi
    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    model.eval()

    result_dir = os.path.join("results", config["name"] + suffix, "test")
    os.makedirs(os.path.join(result_dir, "images"), exist_ok=True)
    metrics_path = os.path.join(result_dir, "metrics.txt")

    # Metryki w przestrzeni [-1,1] -> data_range=2.0
    mse_metric = MeanSquaredError().to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=2.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)

    with torch.no_grad(), open(metrics_path, "w") as f_out:
        f_out.write("idx\tmse\tpsnr\tssim\n")

        for idx, (x, _) in enumerate(val_loader):
            x = x.to(device)  # x w [-1, 1]

            if input_variant == "noisy":
                # szum w przestrzeni wejściowej, clamp do [-1,1]
                x_input = x + noise_std * torch.randn_like(x)
                x_input = x_input.clamp(-1.0, 1.0)
                out = model(x_input)
            elif input_variant == "noisy_latent":
                # szum w przestrzeni latentnej
                z = model.encode(x)
                if isinstance(z, tuple):
                    z_noisy = tuple(lat + noise_std * torch.randn_like(lat) for lat in z)
                    out = model.decode(*z_noisy)
                else:
                    z_noisy = z + noise_std * torch.randn_like(z)
                    out = model.decode(z_noisy)
            else:  # "clean"
                out = model(x)

            # Jeśli model zwraca tuple/listę, bierzemy rekonstrukcję jako pierwszy element
            x_hat = out[0] if isinstance(out, (tuple, list)) else out
            x_hat = x_hat.clamp(-1.0, 1.0)

            # Aktualizacja metryk globalnych
            mse_metric.update(x_hat, x)
            psnr_metric.update(x_hat, x)
            ssim_metric.update(x_hat, x)

            # Per-batch MSE / PSNR / SSIM do logu
            mse_val = torch.nn.functional.mse_loss(x_hat, x).item()
            # PSNR dla data_range = 2.0 -> max_I^2 = 4.0
            psnr_val = 10 * torch.log10(torch.tensor(4.0) / (mse_val + 1e-10)).item()
            # SSIM per batch – osobna instancja, żeby nie mieszać ze średnią
            ssim_batch_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
            ssim_val = ssim_batch_metric(x_hat, x).item()

            f_out.write(f"{idx}\t{mse_val:.6f}\t{psnr_val:.6f}\t{ssim_val:.6f}\n")

        mse_avg = mse_metric.compute().item()
        psnr_avg = psnr_metric.compute().item()
        ssim_avg = ssim_metric.compute().item()

        f_out.write(f"avg\t{mse_avg:.6f}\t{psnr_avg:.6f}\t{ssim_avg:.6f}\n")

    # konsola (opcjonalnie)
    if log:
        print(f"\nEvaluation results ({input_variant}):")
        print(f"  MSE:  {mse_avg:.6f}")
        print(f"  PSNR: {psnr_avg:.6f}")
        print(f"  SSIM: {ssim_avg:.6f}")

    # przykładowe obrazy (rekonstrukcje)
    save_images(
        model,
        val_loader,
        device,
        save_path=os.path.join(result_dir, "images", "examples.png"),
        num_images=10,
        add_noise=(input_variant == "noisy"),
        latent_noise=(input_variant == "noisy_latent"),
        noise_std=noise_std,
    )