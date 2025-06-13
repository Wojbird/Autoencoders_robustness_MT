import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
import torch.nn.functional as F


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_device():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True


def save_metrics(metrics_dict, save_path):
    with open(save_path, "w") as f:
        for k, v in metrics_dict.items():
            if isinstance(v, list):
                f.write(f"{k}: {','.join(f'{x:.6f}' for x in v)}\n")
            else:
                f.write(f"{k}: {v:.6f}\n")


def plot_metrics(metrics_dict, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    for metric in ["mse", "psnr", "ssim"]:
        train_key = f"{metric}_train"
        val_key = f"{metric}_val"

        if train_key in metrics_dict and val_key in metrics_dict:
            plt.figure()
            plt.plot(metrics_dict[train_key], label="Train")
            plt.plot(metrics_dict[val_key], label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel(metric.upper())
            plt.title(f"{metric.upper()} over Epochs")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, f"{metric}.png"))
            plt.close()


def save_images(model, dataloader, device, save_path,
                num_images=8, add_noise=False, latent_noise=False, noise_std=0.1):
    model.eval()
    images_shown = 0
    rows = []

    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            x_vis = x.clone()

            if add_noise:
                x = x + noise_std * torch.randn_like(x)
                x = torch.clamp(x, 0., 1.)

            if latent_noise:
                z = model.encode(x)
                z = z + noise_std * torch.randn_like(z)
                out = model.decode(z)
            else:
                out = model(x)

            for i in range(x.shape[0]):
                if images_shown >= num_images:
                    break

                orig = to_pil_image(x_vis[i].cpu())
                recon = to_pil_image(torch.clamp(out[i].cpu(), 0., 1.))

                pair = make_grid([x_vis[i].cpu(), torch.clamp(out[i].cpu(), 0., 1.)], nrow=2)
                rows.append(pair)
                images_shown += 1

            if images_shown >= num_images:
                break

    grid = make_grid(rows, nrow=1, padding=10)
    plt.figure(figsize=(num_images * 2, 4))
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0))
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()