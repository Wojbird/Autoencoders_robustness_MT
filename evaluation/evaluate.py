import os
import json
import torch
from torch.utils.data import DataLoader

from data.data_setter import get_subnet_datasets, get_imagenet_datasets
from utils.helpers import get_device, save_metrics, save_images
from utils.metrics import (
    calculate_mse, calculate_psnr, calculate_ssim,
    calculate_mse_latent, calculate_psnr_latent, calculate_ssim_latent
)


def evaluate_model(model_class, config_path, input_variant="clean", dataset_variant="subset", log=False):
    # Load configuration
    with open(config_path, "r") as f:
        config = json.load(f)

    device = get_device()

    # Load validation dataset
    if dataset_variant == "subset":
        _, val_set = get_subnet_datasets("datasets/subset_imagenet/", image_size=config["image_size"])
    else:
        _, val_set = get_imagenet_datasets("/raid/kszyc/datasets/ImageNet2012", image_size=config["image_size"])

    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    noise_std = config.get("noise_std", 0.1)

    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    # Initialize model and load weights
    model = model_class(config).to(device)
    model.load_state_dict(torch.load(config["pretrained_path"], map_location=device))
    model.eval()

    # Prepare result path
    subname = {
        "clean": "",
        "noisy": "_noisy",
        "noisy-latent": "_noisy_latent"
    }[input_variant]

    result_dir = os.path.join("results", config["name"] + subname, "test")
    os.makedirs(os.path.join(result_dir, "images"), exist_ok=True)

    # Evaluate and save images
    if input_variant == "clean":
        mse = calculate_mse(model, val_loader, device)
        psnr = calculate_psnr(model, val_loader, device)
        ssim = calculate_ssim(model, val_loader, device)
        save_images(
            model, val_loader, device,
            save_path=os.path.join(result_dir, "images", "examples.png"),
            num_images=10
        )
    elif input_variant == "noisy":
        mse = calculate_mse(model, val_loader, device, add_noise=True, noise_std=noise_std)
        psnr = calculate_psnr(model, val_loader, device, add_noise=True, noise_std=noise_std)
        ssim = calculate_ssim(model, val_loader, device, add_noise=True, noise_std=noise_std)
        save_images(
            model, val_loader, device,
            save_path=os.path.join(result_dir, "images", "examples.png"),
            num_images=10,
            add_noise=True,
            noise_std=noise_std
        )
    elif input_variant == "noisy-latent":
        mse = calculate_mse_latent(model, val_loader, device, noise_std=noise_std)
        psnr = calculate_psnr_latent(model, val_loader, device, noise_std=noise_std)
        ssim = calculate_ssim_latent(model, val_loader, device, noise_std=noise_std)
        save_images(
            model, val_loader, device,
            save_path=os.path.join(result_dir, "images", "examples.png"),
            num_images=10,
            latent_noise=True,
            noise_std=noise_std
        )
    else:
        raise ValueError(f"Unknown input_variant: {input_variant}")

    metrics = {
        "mse_val": [mse],
        "psnr_val": [psnr],
        "ssim_val": [ssim]
    }

    if log:
        print(f"\nEvaluation results ({input_variant}):")
        print(f"  MSE:  {mse:.6f}")
        print(f"  PSNR: {psnr:.2f}")
        print(f"  SSIM: {ssim:.4f}")

    save_metrics(metrics, os.path.join(result_dir, "metrics.txt"))