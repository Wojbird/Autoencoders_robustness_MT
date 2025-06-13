import os
import json
import torch
from torch.utils.data import DataLoader

from data.data_setter import get_subnet_datasets, get_imagenet_datasets
from utils.helpers import get_device, save_metrics, save_images, plot_metrics
from utils.metrics import calculate_mse, calculate_psnr, calculate_ssim


def train_model(model_class, config_path, input_variant="clean", dataset_variant="subset", log=False):
    with open(config_path, "r") as f:
        config = json.load(f)

    device = get_device()

    # Load dataset
    if dataset_variant == "subset":
        train_set, val_set = get_subnet_datasets("datasets/subset_imagenet/", image_size=config["image_size"])
    else:
        train_set, val_set = get_imagenet_datasets("datasets/full_imagenet/", image_size=config["image_size"])

    batch_size = config["batch_size"]
    num_workers = config["num_workers"]

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    # Prepare model, optimizer, loss
    model = model_class(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    criterion = torch.nn.MSELoss()

    # Prepare result directory
    result_dir = os.path.join("results", config["name"], "training")
    os.makedirs(os.path.join(result_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(result_dir, "plots"), exist_ok=True)

    # Training history
    history = {key: [] for key in ["mse_train", "mse_val", "psnr_train", "psnr_val", "ssim_train", "ssim_val"]}

    # Training loop
    for epoch in range(config["epochs"]):
        model.train()
        for i, (x, _) in enumerate(train_loader):
            x = x.to(device)
            output = model(x)
            loss = criterion(output, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if log and i % max(1, len(train_loader) // 10) == 0:
                print(f"[Epoch {epoch+1}/{config['epochs']}] Batch {i}/{len(train_loader)} – Loss: {loss.item():.4f}")

        # Validation metrics
        model.eval()
        with torch.no_grad():
            mse_train = calculate_mse(model, train_loader, device)
            psnr_train = calculate_psnr(model, train_loader, device)
            ssim_train = calculate_ssim(model, train_loader, device)

            mse_val = calculate_mse(model, val_loader, device)
            psnr_val = calculate_psnr(model, val_loader, device)
            ssim_val = calculate_ssim(model, val_loader, device)

        # Save metrics
        for k, v in zip(history.keys(), [mse_train, mse_val, psnr_train, psnr_val, ssim_train, ssim_val]):
            history[k].append(v)

        if log:
            print(f"[Epoch {epoch+1}] MSE: {mse_val:.4f}, PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}")

        save_images(model, val_loader, device,
                    save_path=os.path.join(result_dir, "images", f"epoch_{epoch+1}.png"),
                    num_images=4)

    save_metrics(history, os.path.join(result_dir, "metrics.txt"))
    plot_metrics(history, os.path.join(result_dir, "plots"))

    print("Training with clean input complete.")