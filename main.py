import json
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
from torch.utils.data import DataLoader
import argparse
import importlib
import sys
import logging
from pathlib import Path
import matplotlib

torch.set_num_threads(1)

matplotlib.use("Agg")

from training.train_clean import train_clean_model
from training.train_noisy import train_noisy_model
from training.train_noisy_latent import train_noisy_latent_model
from evaluation.evaluate import evaluate_model
from utils.helpers import set_seed, setup_device, get_device
from data.data_setter import get_subnet_datasets, get_imagenet_datasets


def parse_args():
    parser = argparse.ArgumentParser(description="Autoencoder Robustness Framework")

    parser.add_argument("--mode", choices=["train", "test", "train_test"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--type", choices=["clean", "noisy", "noisy_latent", "all"], default="clean")
    parser.add_argument("--input", choices=["subset", "full"], default="subset")
    parser.add_argument(
        "--eval_on",
        choices=["clean", "noisy", "noisy_latent", "all"],
        default="all",
        help="Evaluation condition used in test mode."
    )
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--log_wandb", action="store_true")
    parser.add_argument("--gpu", type=int, default=None)

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

    return DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


MAIN_EXPERIMENT_SIZES = {16, 32, 64, 128, 256}


def _model_size_from_stem(stem: str) -> int | None:
    suffix = stem.split("_")[-1]
    if suffix.isdigit():
        return int(suffix)
    return None


def discover_models(base_dir: str, target: str) -> list:
    base = Path(base_dir)

    def is_real_model_file(p: Path) -> bool:
        parts = set(p.parts)
        return (
            p.name.endswith(".py")
            and not p.name.endswith("_base.py")
            and p.name != "__init__.py"
            and "junk_folder" not in parts
        )

    def is_main_experiment_model_file(p: Path) -> bool:
        if not is_real_model_file(p):
            return False
        if "test" in set(p.parts):
            return False

        size = _model_size_from_stem(p.stem)
        return size in MAIN_EXPERIMENT_SIZES

    if target == "all":
        return [p for p in base.rglob("*.py") if is_main_experiment_model_file(p)]

    exact_matches = [p for p in base.rglob(f"{target}.py") if is_real_model_file(p)]
    if exact_matches:
        return exact_matches

    candidate_dir = base / target
    if candidate_dir.is_dir():
        return [p for p in candidate_dir.glob("*.py") if is_main_experiment_model_file(p)]

    print(f"Error: Could not find model '{target}' in {base_dir}")
    sys.exit(1)


def import_model(module_path: Path):
    rel_path = module_path.with_suffix("").relative_to(Path("models"))
    module_name = ".".join(["models"] + list(rel_path.parts))
    module = importlib.import_module(module_name)

    model_class = getattr(module, "model_class", None)
    config_path = getattr(module, "config_path", None)

    if model_class is None or config_path is None:
        print(f"Error: Module {module_name} must define 'model_class' and 'config_path'")
        sys.exit(1)

    return model_class, config_path


def is_vq_config(cfg: dict) -> bool:
    return str(cfg.get("family", "")).lower() == "vqv"


def run(mode, model_path: Path, input_type, dataset_type, log, log_wandb, gpu_id=None, eval_on="all"):
    model_class, config_path = import_model(model_path)
    cfg = load_config(config_path)

    if input_type == "all":
        variants = ["clean", "noisy", "noisy_latent"]
        if is_vq_config(cfg):
            variants = ["clean", "noisy"]

        for variant in variants:
            print(f"\n--- Running {mode} for input type: {variant} ---")
            run(mode, model_path, variant, dataset_type, log, log_wandb, gpu_id=gpu_id, eval_on=eval_on)
        return

    if input_type == "noisy_latent" and is_vq_config(cfg):
        print(f"Skipping noisy_latent for VQ model: {cfg.get('name', model_path.stem)}")
        return

    model_name = str(cfg.get("name", model_path.stem))
    train_noise_std = (
        float(cfg.get("noise_latent", cfg.get("noise_std", 0.0)))
        if input_type == "noisy_latent"
        else float(cfg.get("noise_std", 0.0))
    )

    results_dir = os.path.join("results", model_name, dataset_type, input_type)
    os.makedirs(results_dir, exist_ok=True)

    ckpt_dir = os.path.join("checkpoints", model_name, dataset_type, input_type)
    ckpt_path = os.path.join(ckpt_dir, f"{model_name}_{input_type}_best.pt")

    trained_ckpt = None

    if mode in ("train", "train_test"):
        if input_type == "clean":
            trained_ckpt = train_clean_model(
                model_class,
                config_path,
                dataset_type=dataset_type,
                log=log,
                log_wandb=log_wandb,
                gpu_id=gpu_id,
            )
        elif input_type == "noisy":
            trained_ckpt = train_noisy_model(
                model_class,
                config_path,
                dataset_type=dataset_type,
                log=log,
                log_wandb=log_wandb,
                gpu_id=gpu_id,
            )
        elif input_type == "noisy_latent":
            trained_ckpt = train_noisy_latent_model(
                model_class,
                config_path,
                dataset_type=dataset_type,
                log=log,
                log_wandb=log_wandb,
                gpu_id=gpu_id,
            )
        else:
            raise ValueError(f"Unknown input type: {input_type}")

        if isinstance(trained_ckpt, str) and trained_ckpt:
            ckpt_path = trained_ckpt

        print(f"Best checkpoint: {ckpt_path}")

    if mode in ("test", "train_test"):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}\n"
                f"Run training first or verify your trainer saves the best checkpoint there."
            )

        device = get_device(gpu_id)

        try:
            model = model_class(cfg).to(device)
        except TypeError:
            model = model_class().to(device)

        state = torch.load(ckpt_path, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)

        val_loader = build_val_loader(cfg, dataset_type)

        eval_noise_std = float(cfg.get("eval_noise_std", cfg.get("noise_std", train_noise_std)))
        eval_variants = ["clean", "noisy", "noisy_latent"] if eval_on == "all" else [eval_on]

        if is_vq_config(cfg):
            eval_variants = [v for v in eval_variants if v != "noisy_latent"]

        for eval_variant in eval_variants:
            test_dir = os.path.join(results_dir, "test", f"{eval_variant}_eval")
            metrics = evaluate_model(
                model=model,
                dataloader=val_loader,
                device=device,
                variant=eval_variant,
                noise_std=eval_noise_std,
                results_dir=test_dir,
                noise_seed=1234,
            )

            print(f"Evaluation metrics ({eval_variant}):")
            for k, v in metrics.items():
                print(f"  {k}: {v:.6f}")


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO if args.log else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    set_seed(42)
    setup_device(args.gpu)

    model_paths = discover_models("models", args.model)

    for path in model_paths:
        print(f"\n>>> Running {args.mode} for model: {path}")
        run(
            args.mode,
            path,
            args.type,
            args.input,
            args.log,
            args.log_wandb,
            gpu_id=args.gpu,
            eval_on=args.eval_on,
        )


if __name__ == "__main__":
    main()