import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def setup_device(deterministic: bool = False, num_threads: int = 4):
    torch.set_num_threads(num_threads)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_metrics(metrics_dict, save_path):
    with open(save_path, "w") as f:
        for k, v in metrics_dict.items():
            if isinstance(v, list):
                f.write(f"{k}: {','.join(f'{x:.6f}' for x in v)}\n")
            else:
                f.write(f"{k}: {v}\n")


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
                num_images=8, add_noise=False, latent_noise=False, noise_std=0.1, seed=None):
    model.eval()

    g = None
    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(int(seed))

    images_shown = 0
    images_orig = []
    images_recon = []

    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            x_vis = x.clone()

            if add_noise:
                if g is None:
                    noise = torch.randn_like(x) * noise_std
                else:
                    noise = torch.randn(x.shape, device=x.device, generator=g) * noise_std
                x = torch.clamp(x + noise, -1., 1.)

            if latent_noise:
                z = model.encode(x)
                if g is None:
                    eps = torch.randn_like(z) * noise_std
                else:
                    eps = torch.randn(z.shape, device=z.device, generator=g) * noise_std
                out = model.decode(z + eps)
            else:
                out = model(x)

            out_tensor = out[0] if isinstance(out, (tuple, list)) else out
            out_tensor = out_tensor.clamp(-1., 1.).cpu()
            x_vis = x_vis.cpu()

            for i in range(min(x_vis.shape[0], num_images - images_shown)):
                images_orig.append(x_vis[i])
                images_recon.append(out_tensor[i])
                images_shown += 1

            if images_shown >= num_images:
                break

    images_orig = [(img.clamp(-1., 1.) + 1) / 2 for img in images_orig]
    images_recon = [(img.clamp(-1., 1.) + 1) / 2 for img in images_recon]

    pairs = [torch.stack([images_orig[i], images_recon[i]]) for i in range(images_shown)]
    all_images = torch.cat(pairs, dim=0)
    grid = make_grid(all_images, nrow=2, padding=10)

    plt.figure(figsize=(num_images * 2, 4))
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0))
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()