import argparse
import importlib
import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")

from training.train_clean import train_model as train_clean_model
from training.train_noisy import train_model as train_noisy_model
from training.train_noisy_latent import train_model as train_noisy_latent_model
from evaluation.evaluate import evaluate_model
from utils.helpers import set_seed, setup_device


def parse_args():
    parser = argparse.ArgumentParser(description="Autoencoder Robustness Framework")

    parser.add_argument("--mode", choices=["train", "test", "train&test"], required=True,
                        help="Operation mode")
    parser.add_argument("--model", required=True,
                        help="Model name (e.g., ae_convtranspose512_7x7), group (e.g., conv), or 'all'")
    parser.add_argument("--type", choices=["clean", "noisy", "noisy-latent"], default="clean",
                        help="Input variant")
    parser.add_argument("--input", choices=["subset", "full"], default="subset",
                        help="Dataset source: subset or full")
    parser.add_argument("--log", action="store_true",
                        help="Enable detailed logging")

    return parser.parse_args()


def discover_models(base_dir: str, target: str) -> list:
    base = Path(base_dir)

    if target == "all":
        return [p for p in base.rglob("*.py") if p.name != "__init__.py"]

    potential_file = list(base.rglob(f"{target}.py"))
    if potential_file:
        return potential_file

    if (base / target).is_dir():
        return list((base / target).glob("*.py"))

    print(f"Error: Could not find model '{target}' in {base_dir}")
    sys.exit(1)


def import_model(module_path: Path):
    # e.g., models/conv/ae_convtranspose512_7x7.py → models.conv.ae_convtranspose512_7x7
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
    model_class, config_path = import_model(model_path)

    if mode in ("train", "train&test"):
        if input_type == "clean":
            train_clean_model(model_class, config_path, input_variant=input_type,
                              dataset_variant=dataset_type, log=log)
        elif input_type == "noisy":
            train_noisy_model(model_class, config_path, input_variant=input_type,
                              dataset_variant=dataset_type, log=log)
        elif input_type == "noisy-latent":
            train_noisy_latent_model(model_class, config_path, input_variant=input_type,
                                     dataset_variant=dataset_type, log=log)
        else:
            raise ValueError(f"Unknown input type: {input_type}")

    if mode in ("test", "train&test"):
        evaluate_model(model_class, config_path,
                       input_variant=input_type,
                       dataset_variant=dataset_type,
                       log=log)


def main():
    args = parse_args()
    set_seed(42)
    setup_device()

    model_paths = discover_models("models", args.model)

    for path in model_paths:
        print(f"\n>>> Running {args.mode} for model: {path}")
        run(args.mode, path, args.type, args.input, args.log)


if __name__ == "__main__":
    main()