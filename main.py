from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Mapping
from pathlib import Path


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib
import torch
from torch.utils.data import DataLoader


matplotlib.use("Agg")

from data.data_setter import get_imagenet_datasets, get_subnet_datasets
from evaluation.evaluate import evaluate_model
from robustness.model_registry import ModelSpec, discover_model_specs
from training.train_clean import train_clean_model
from training.train_noisy import train_noisy_model
from training.train_noisy_latent import train_noisy_latent_model
from utils.helpers import get_device, set_seed, setup_device


PROJECT_ROOT = Path(__file__).resolve().parent
TRAINERS = {
    "clean": train_clean_model,
    "noisy": train_noisy_model,
    "noisy_latent": train_noisy_latent_model,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the canonical autoencoder experiment matrix."
    )
    parser.add_argument(
        "--mode",
        choices=("train", "test", "train_test"),
        required=True,
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Exact model name, canonical family/alias, or 'all'.",
    )
    parser.add_argument(
        "--type",
        choices=("clean", "noisy", "noisy_latent", "all"),
        default="clean",
        help="Training/checkpoint variant.",
    )
    parser.add_argument(
        "--input",
        choices=("subset", "full"),
        default="subset",
        help="Dataset scope retained for backward CLI compatibility.",
    )
    parser.add_argument(
        "--eval_on",
        choices=("clean", "noisy", "noisy_latent", "all"),
        default="all",
        help="Evaluation condition used in test modes.",
    )
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--log_wandb", action="store_true")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Request deterministic PyTorch algorithms where available.",
    )
    return parser.parse_args()


def _build_validation_loader(config: dict, dataset_type: str) -> DataLoader:
    image_size = int(config.get("image_size", 224))
    batch_size = int(config.get("batch_size", 32))
    num_workers = int(config.get("num_workers", 4))
    split_seed = int(config.get("dataset_split_seed", 42))

    if dataset_type == "subset":
        _, validation_set = get_subnet_datasets(
            image_size=image_size,
            val_split=0.1,
            split_seed=split_seed,
        )
    elif dataset_type == "full":
        _, validation_set = get_imagenet_datasets(image_size=image_size)
    else:
        raise ValueError("dataset_type must be 'subset' or 'full'.")

    pin_memory = torch.cuda.is_available()
    return DataLoader(
        validation_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def _load_checkpoint_state(checkpoint_path: Path, device: torch.device) -> Mapping:
    try:
        payload = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        # Compatibility with trusted, project-generated checkpoints on older
        # PyTorch releases that do not expose the weights_only argument.
        payload = torch.load(checkpoint_path, map_location=device)

    if not isinstance(payload, Mapping):
        raise TypeError(
            f"Checkpoint root must be a mapping, received {type(payload).__name__}: "
            f"{checkpoint_path}"
        )

    if "model_state_dict" in payload:
        payload = payload["model_state_dict"]

    if not isinstance(payload, Mapping):
        raise TypeError(f"model_state_dict is not a mapping: {checkpoint_path}")
    return payload


def _variants_for_spec(spec: ModelSpec, requested: str) -> tuple[str, ...]:
    supported = spec.supported_training_variants
    if requested == "all":
        return supported
    if requested not in supported:
        raise ValueError(
            f"{spec.model_name} does not support training variant {requested!r}. "
            f"Supported variants: {supported}."
        )
    return (requested,)


def _evaluation_variants(spec: ModelSpec, requested: str) -> tuple[str, ...]:
    supported = spec.supported_training_variants
    if requested == "all":
        return supported
    if requested not in supported:
        raise ValueError(
            f"{spec.model_name} does not support evaluation variant {requested!r}. "
            f"Supported variants: {supported}."
        )
    return (requested,)


def _checkpoint_path(
    *,
    spec: ModelSpec,
    dataset_type: str,
    training_variant: str,
) -> Path:
    return spec.checkpoint_path(
        PROJECT_ROOT / "checkpoints",
        dataset_type,
        training_variant,
    )


def run_variant(
    *,
    mode: str,
    spec: ModelSpec,
    training_variant: str,
    dataset_type: str,
    evaluate_on: str,
    log: bool,
    log_wandb: bool,
    gpu_id: int | None,
) -> None:
    config = spec.config
    experiment_seed = int(config.get("experiment_seed", 42))
    set_seed(experiment_seed)

    checkpoint_path = _checkpoint_path(
        spec=spec,
        dataset_type=dataset_type,
        training_variant=training_variant,
    )

    if mode in {"train", "train_test"}:
        trainer = TRAINERS[training_variant]
        trained_path = trainer(
            spec.model_class,
            str(spec.config_path),
            dataset_type=dataset_type,
            log=log,
            log_wandb=log_wandb,
            gpu_id=gpu_id,
        )
        checkpoint_path = Path(trained_path).resolve()
        print(f"Best checkpoint: {checkpoint_path}")

    if mode not in {"test", "train_test"}:
        return

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Train this exact "
            "model/dataset/variant combination first."
        )

    device = get_device(gpu_id)
    model = spec.model_class(config).to(device)
    state_dict = _load_checkpoint_state(checkpoint_path, device)
    model.load_state_dict(state_dict, strict=True)

    validation_loader = _build_validation_loader(config, dataset_type)
    eval_noise_std = float(
        config.get("eval_noise_std", config.get("noise_std", 0.1))
    )
    eval_noise_seed = int(config.get("eval_noise_seed", 1234))
    latent_noise_relative = bool(config.get("latent_noise_relative", True))

    for evaluation_variant in _evaluation_variants(spec, evaluate_on):
        output_dir = (
            PROJECT_ROOT
            / "results"
            / spec.model_name
            / dataset_type
            / training_variant
            / "test"
            / f"{evaluation_variant}_eval"
        )
        metrics = evaluate_model(
            model=model,
            dataloader=validation_loader,
            device=device,
            variant=evaluation_variant,
            noise_std=(
                float(config.get("noise_latent", eval_noise_std))
                if evaluation_variant == "noisy_latent"
                else eval_noise_std
            ),
            results_dir=str(output_dir),
            noise_seed=eval_noise_seed,
            latent_noise_relative=latent_noise_relative,
        )

        print(
            f"{spec.model_name} | trained={training_variant} | "
            f"evaluated={evaluation_variant}"
        )
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.6f}")


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.log else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    os.chdir(PROJECT_ROOT)
    setup_device(args.gpu, deterministic=args.deterministic)

    specs = discover_model_specs(
        PROJECT_ROOT,
        selectors=[args.model],
    )
    if not specs:
        raise ValueError(
            f"No canonical model matched {args.model!r}. Use an exact model "
            "name, a registered family alias, or 'all'."
        )

    for spec in specs:
        for training_variant in _variants_for_spec(spec, args.type):
            print(
                f"\n>>> {args.mode}: {spec.model_name} "
                f"[{args.input}/{training_variant}]"
            )
            run_variant(
                mode=args.mode,
                spec=spec,
                training_variant=training_variant,
                dataset_type=args.input,
                evaluate_on=args.eval_on,
                log=args.log,
                log_wandb=args.log_wandb,
                gpu_id=args.gpu,
            )


if __name__ == "__main__":
    main()
