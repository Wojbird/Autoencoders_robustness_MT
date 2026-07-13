from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch


PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from robustness.attacks import (
    SUPPORTED_ATTACKS,
    build_attack_conditions,
)
from robustness.data import build_evaluation_loader
from robustness.evaluator import evaluate_checkpoint
from robustness.model_registry import discover_model_specs
from robustness.reporting import build_reports
from utils.helpers import set_seed


TRAINING_VARIANTS = (
    "clean",
    "noisy",
    "noisy_latent",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate trained autoencoders on the local ImageNet subset "
            "using one CUDA GPU."
        )
    )

    parser.add_argument(
        "--mode",
        choices=(
            "all",
            "evaluate",
            "report",
        ),
        default="all",
        help="Run evaluation, reporting, or both.",
    )

    parser.add_argument(
        "--model",
        nargs="+",
        default=["all"],
        help=(
            "all, a family (conv/unet/adversarial/vqv), "
            "or one or more exact model names."
        ),
    )

    parser.add_argument(
        "--training-variant",
        nargs="+",
        default=["all"],
        help="all, clean, noisy, or noisy_latent.",
    )

    parser.add_argument(
        "--attacks",
        nargs="+",
        default=["all"],
        help=(
            "all or any of: "
            + ", ".join(SUPPORTED_ATTACKS)
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Override configured batch size. Default is 8 for an RTX 3070 "
            "with 8 GB VRAM."
        ),
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help=(
            "Override DataLoader workers. Default is 2 for Windows."
        ),
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Override configured sample count. Use 0 for the entire "
            "deterministic validation part of the subset."
        ),
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/robustness_experiments.json"
        ),
    )

    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("checkpoints"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/robustness"),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute result CSV files that already exist.",
    )

    parser.add_argument(
        "--strict-checkpoints",
        action="store_true",
        help=(
            "Stop immediately when an expected checkpoint is missing. "
            "Do not use this for the current 54-checkpoint experiment."
        ),
    )

    parser.add_argument(
        "--save-previews",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    parser.add_argument(
        "--preview-count",
        type=int,
        default=None,
    )

    return parser.parse_args()


def _resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def load_config(path: Path) -> dict:
    resolved_path = _resolve_project_path(path)

    with resolved_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_training_variants(
    values: list[str],
) -> list[str]:
    normalised = [
        value.strip().lower()
        for value in values
    ]

    if "all" in normalised:
        return list(TRAINING_VARIANTS)

    unknown = set(normalised).difference(TRAINING_VARIANTS)

    if unknown:
        raise ValueError(
            "Unknown training variants: "
            f"{sorted(unknown)}. Supported variants: "
            f"{list(TRAINING_VARIANTS)}"
        )

    return normalised


def setup_single_gpu() -> torch.device:
    """
    Use only cuda:0. No CPU fallback and no multi-GPU execution are allowed.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this experiment, but "
            "torch.cuda.is_available() returned False."
        )

    gpu_count = torch.cuda.device_count()

    if gpu_count < 1:
        raise RuntimeError(
            "No CUDA GPU is visible to PyTorch."
        )

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(0)

    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True

    total_memory_gib = properties.total_memory / (1024**3)

    print("CUDA is available: True")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count visible to PyTorch: {gpu_count}")

    print(
        "Using only [0] "
        f"{properties.name} | "
        f"{total_memory_gib:.1f} GB | "
        "compute capability "
        f"{properties.major}.{properties.minor}"
    )

    if gpu_count > 1:
        print(
            "[warning] More than one GPU is visible, but this program will "
            "use only cuda:0 and will not start parallel workers."
        )

    return device


def write_missing_checkpoints(
    rows: list[dict],
    output_dir: Path,
) -> None:
    summary_directory = output_dir / "summary"
    path = summary_directory / "missing_checkpoints.csv"

    if not rows:
        if path.exists():
            path.unlink()

        return

    summary_directory.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "model_name",
                "family",
                "latent_channels",
                "latent_elements",
                "training_variant",
                "checkpoint_dataset",
                "evaluation_dataset",
                "checkpoint_path",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"Missing-checkpoint report: {path}")


def validate_experiment_config(config: dict) -> tuple[str, str]:
    checkpoint_dataset = str(
        config.get("checkpoint_dataset", "full")
    ).lower()

    evaluation_dataset = str(
        config.get("evaluation_dataset", "subset")
    ).lower()

    if checkpoint_dataset != "full":
        raise ValueError(
            "The available checkpoints are stored under dataset='full'. "
            "Set checkpoint_dataset to 'full'."
        )

    if evaluation_dataset != "subset":
        raise ValueError(
            "This experiment is restricted to "
            "evaluation_dataset='subset'."
        )

    return checkpoint_dataset, evaluation_dataset


def run_evaluation(
    args: argparse.Namespace,
    config: dict,
) -> None:
    checkpoint_dataset, _ = validate_experiment_config(config)

    experiment_seed = int(config.get("seed", 1234))
    set_seed(experiment_seed)
    device = setup_single_gpu()

    model_specs = discover_model_specs(
        project_root=PROJECT_ROOT,
        selectors=args.model,
    )

    training_variants = resolve_training_variants(
        args.training_variant
    )

    conditions = build_attack_conditions(
        config=config,
        selected=args.attacks,
    )

    image_sizes = {
        int(spec.config.get("image_size", 224))
        for spec in model_specs
    }

    if len(image_sizes) != 1:
        raise ValueError(
            "Selected models use different image sizes: "
            f"{sorted(image_sizes)}"
        )

    image_size = next(iter(image_sizes))
    dataset_config = config.get("dataset", {})
    evaluation_config = config.get("evaluation", {})

    configured_device = str(
        evaluation_config.get("device", "cuda:0")
    ).lower()

    if configured_device != "cuda:0":
        raise ValueError(
            "evaluation.device must be 'cuda:0' for this experiment."
        )

    batch_size = (
        int(args.batch_size)
        if args.batch_size is not None
        else int(evaluation_config.get("batch_size", 8))
    )

    num_workers = (
        int(args.num_workers)
        if args.num_workers is not None
        else int(evaluation_config.get("num_workers", 2))
    )

    max_samples = (
        int(args.max_samples)
        if args.max_samples is not None
        else int(dataset_config.get("max_samples", 1000))
    )

    save_previews = (
        bool(args.save_previews)
        if args.save_previews is not None
        else bool(evaluation_config.get("save_previews", False))
    )

    preview_count = (
        int(args.preview_count)
        if args.preview_count is not None
        else int(evaluation_config.get("preview_count", 4))
    )

    output_directory = _resolve_project_path(args.output_dir)
    checkpoint_root = _resolve_project_path(args.checkpoint_root)

    subset_root = _resolve_project_path(
        Path(
            str(
                dataset_config.get(
                    "subset_root",
                    "datasets/subset_imagenet/",
                )
            )
        )
    )

    dataloader, manifest = build_evaluation_loader(
        subset_root=subset_root,
        image_size=image_size,
        validation_fraction=float(
            dataset_config.get("validation_fraction", 0.1)
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        max_samples=max_samples,
        split_seed=int(dataset_config.get("split_seed", 42)),
        sample_seed=int(dataset_config.get("sample_seed", 2026)),
        manifest_path=output_directory / "run_manifest.json",
        device=device,
        checkpoint_dataset=checkpoint_dataset,
    )

    print(
        "Evaluation dataset: subset | "
        f"selected images: {manifest['requested_samples']} | "
        f"validation pool: {manifest['validation_length']} | "
        f"batch size: {batch_size} | "
        f"image size: {image_size} | "
        "normalization: [0, 1]"
    )

    print(
        f"Selected model configurations: {len(model_specs)} | "
        f"executable conditions per checkpoint: {len(conditions)}"
    )

    missing_checkpoints: list[dict] = []
    evaluated_checkpoints = 0

    for spec in model_specs:
        if spec.uses_checkpoint_override:
            print(
                f"[override] {spec.model_name}: "
                f"{spec.config_override}"
            )

        for training_variant in training_variants:
            if training_variant not in spec.supported_training_variants:
                print(
                    f"[skip] {spec.model_name} has no "
                    f"'{training_variant}' training variant."
                )

                continue

            checkpoint_path = spec.checkpoint_path(
                checkpoint_root=checkpoint_root,
                checkpoint_dataset=checkpoint_dataset,
                training_variant=training_variant,
            )

            if not checkpoint_path.exists():
                missing_row = {
                    "model_name": spec.model_name,
                    "family": spec.family,
                    "latent_channels": spec.latent_channels,
                    "latent_elements": spec.latent_elements,
                    "training_variant": training_variant,
                    "checkpoint_dataset": checkpoint_dataset,
                    "evaluation_dataset": "subset",
                    "checkpoint_path": str(checkpoint_path),
                }

                missing_checkpoints.append(missing_row)

                message = f"Checkpoint not found: {checkpoint_path}"

                if args.strict_checkpoints:
                    raise FileNotFoundError(message)

                print(f"[missing] {message}")
                continue

            print(
                "\n=== "
                f"{spec.model_name} | "
                f"{training_variant} | "
                f"{len(conditions)} conditions ==="
            )

            evaluate_checkpoint(
                spec=spec,
                training_variant=training_variant,
                checkpoint_dataset=checkpoint_dataset,
                checkpoint_path=checkpoint_path,
                dataloader=dataloader,
                conditions=conditions,
                device=device,
                output_root=output_directory,
                attack_seed=experiment_seed,
                overwrite=args.overwrite,
                save_previews=save_previews,
                preview_count=preview_count,
            )

            evaluated_checkpoints += 1

    write_missing_checkpoints(
        missing_checkpoints,
        output_directory,
    )

    print(
        "Evaluation completed for "
        f"{evaluated_checkpoints} available checkpoints."
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_directory = _resolve_project_path(args.output_dir)

    if args.mode in {"all", "evaluate"}:
        run_evaluation(args, config)

    if args.mode in {"all", "report"}:
        validate_experiment_config(config)

        summary_path = build_reports(
            output_root=output_directory,
            overwrite_derived=True,
        )

        print(f"Summary: {summary_path}")
        print(f"Plots: {output_directory / 'plots'}")


if __name__ == "__main__":
    main()