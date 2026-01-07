import json
import os
import torch
from torch.utils.data import DataLoader
import argparse
import importlib
import sys
import logging
from pathlib import Path
import matplotlib
matplotlib.use("Agg")

from training.train_clean import train_clean_model
from training.train_noisy import train_noisy_model
from training.train_noisy_latent import train_noisy_latent_model
from evaluation.evaluate import evaluate_model
from utils.helpers import set_seed, setup_device
from data.data_setter import get_subnet_datasets, get_imagenet_datasets
from utils.helpers import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Autoencoder Robustness Framework")

    parser.add_argument("--mode", choices=["train", "test", "train_test"], required=True,
                        help="Operation mode")
    parser.add_argument("--model", required=True,
                        help="Model name (e.g., ae_convtranspose512_7x7), group (e.g., conv), or 'all'")
    parser.add_argument("--type", choices=["clean", "noisy", "noisy_latent", "all"], default="clean",
                        help="Input variant: clean, noisy, noisy_latent or all (runs all three)")
    parser.add_argument("--input", choices=["subset", "full"], default="subset",
                        help="Dataset source: subset or full")
    parser.add_argument("--log", action="store_true",
                        help="Enable detailed logging")

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_val_loader(cfg: dict, dataset_type: str):
    image_size = int(cfg.get("image_size", 224))
    batch_size = int(cfg.get("batch_size", 32))
    num_workers = int(cfg.get("num_workers", 4))

    if dataset_type == "subset":
        _, val_set = get_subnet_datasets(image_size=image_size, val_split=0.1)
    elif dataset_type == "full":
        _, val_set = get_imagenet_datasets(image_size=image_size)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return val_loader


def discover_models(base_dir: str, target: str) -> list:
    base = Path(base_dir)

    if target == "all":
        return [p for p in base.rglob("*.py")]

    potential_file = list(base.rglob(f"{target}.py"))
    if potential_file:
        return potential_file

    if (base / target).is_dir():
        return list((base / target).glob("*.py"))

    print(f"Error: Could not find model '{target}' in {base_dir}")
    sys.exit(1)


def import_model(module_path: Path):
    rel_path = module_path.with_suffix('').relative_to(Path("models"))
    module_name = ".".join(["models"] + list(rel_path.parts))
    module = importlib.import_module(module_name)

    model_class = getattr(module, "model_class", None)
    config_path = getattr(module, "config_path", None)

    if model_class is None or config_path is None:
        print(f"Error: Module {module_name} must define 'model_class' and 'config_path'")
        sys.exit(1)

    return model_class, config_path


def run(mode, model_path: Path, input_type, dataset_type, log):
    if input_type == "all":
        for variant in ["clean", "noisy", "noisy_latent"]:
            print(f"\n--- Running {mode} for input type: {variant} ---")
            run(mode, model_path, variant, dataset_type, log)
        return

    model_class, config_path = import_model(model_path)
    cfg = load_config(config_path)

    model_name = str(cfg.get("name", model_path.stem))
    noise_std = float(cfg.get("noise_std", 0.0))

    # results/<name>/<subset|full>/<variant>/
    results_dir = os.path.join("results", model_name, dataset_type, input_type)
    os.makedirs(results_dir, exist_ok=True)

    # expected "best" checkpoint path
    ckpt_path = os.path.join(results_dir, f"{model_name}_{input_type}_best.pt")

    # ---------- TRAIN ----------
    trained_ckpt = None
    if mode in ("train", "train_test"):
        if input_type == "clean":
            trained_ckpt = train_clean_model(model_class, config_path, dataset_type=dataset_type, log=log)
        elif input_type == "noisy":
            trained_ckpt = train_noisy_model(model_class, config_path, dataset_type=dataset_type, log=log)
        elif input_type == "noisy_latent":
            trained_ckpt = train_noisy_latent_model(model_class, config_path, dataset_type=dataset_type, log=log)
        else:
            raise ValueError(f"Unknown input type: {input_type}")

        # If trainer returns path -> use it; otherwise fall back to expected path
        if isinstance(trained_ckpt, str) and len(trained_ckpt) > 0:
            ckpt_path = trained_ckpt

        print(f"Best checkpoint: {ckpt_path}")

    # ---------- TEST ----------
    if mode in ("test", "train_test"):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}\n"
                f"Run training first or verify your trainer saves the best checkpoint there."
            )

        device = get_device()
        try:
            model = model_class(cfg).to(device)
        except TypeError:
            # Backward compatibility: models without config in __init__
            model = model_class().to(device)

        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)

        val_loader = build_val_loader(cfg, dataset_type)

        test_dir = os.path.join(results_dir, "test")
        metrics = evaluate_model(
            model=model,
            dataloader=val_loader,
            device=device,
            variant=input_type,
            noise_std=noise_std,
            results_dir=test_dir,
        )

        print("Evaluation metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.6f}")


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO if args.log else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    set_seed(42)
    setup_device()

    model_paths = discover_models("models", args.model)

    for path in model_paths:
        print(f"\n>>> Running {args.mode} for model: {path}")
        run(args.mode, path, args.type, args.input, args.log)


if __name__ == "__main__":
    main()