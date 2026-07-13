from __future__ import annotations

import csv
import gc
import os
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from .attacks import (
    AttackCondition,
    apply_input_attack,
    apply_latent_attack,
)
from .metrics import compute_evaluation_metrics
from .model_registry import ModelSpec, load_trained_model


RAW_COLUMNS = [
    "model_name",
    "family",
    "latent_channels",
    "latent_elements",
    "training_variant",
    "checkpoint_dataset",
    "evaluation_dataset",
    "checkpoint_path",
    "attack",
    "severity",
    "direction",
    "attack_space",
    "input_changed",
    "sample_index",
    "sample_path",
    "class_id",
    "input_mse",
    "input_psnr",
    "input_ssim",
    "recon_mse",
    "recon_psnr",
    "recon_ssim",
    "fidelity_mse",
    "fidelity_psnr",
    "fidelity_ssim",
]


def _unwrap_tensor(value):
    if isinstance(value, (tuple, list)):
        return value[0]

    return value


def _encode_decode_with_latent_attack(
    model: torch.nn.Module,
    x: torch.Tensor,
    condition: AttackCondition,
    *,
    attack_seed: int,
    batch_index: int,
) -> torch.Tensor:
    if (
        not hasattr(model, "encode")
        or not callable(getattr(model, "encode"))
        or not hasattr(model, "decode")
        or not callable(getattr(model, "decode"))
    ):
        raise AttributeError(
            f"{model.__class__.__name__} must expose encode() and "
            "decode() for latent-space evaluation."
        )

    latent = _unwrap_tensor(model.encode(x))

    attacked_latent = apply_latent_attack(
        latent,
        condition,
        base_seed=attack_seed,
        batch_index=batch_index,
    )

    return _unwrap_tensor(model.decode(attacked_latent))


def _tensor_to_numpy_image(
    tensor: torch.Tensor,
) -> np.ndarray:
    return (
        tensor.detach()
        .clamp(min=0.0, max=1.0)
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )


def _save_preview(
    path: Path,
    originals: torch.Tensor,
    attacked_inputs: torch.Tensor,
    reconstructions: torch.Tensor,
    *,
    title: str,
    max_images: int,
    clean_reconstructions: torch.Tensor | None = None,
) -> None:
    count = min(max_images, originals.shape[0])

    if count <= 0:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(
        count,
        4,
        figsize=(12, 3 * count),
        squeeze=False,
    )

    for row in range(count):
        if clean_reconstructions is None:
            second_image = attacked_inputs[row]

            labels = (
                "Original",
                "Attacked input",
                "Reconstruction",
                "Absolute error",
            )
        else:
            second_image = clean_reconstructions[row]

            labels = (
                "Original",
                "Clean reconstruction",
                "Attacked reconstruction",
                "Absolute error",
            )

        images = (
            originals[row],
            second_image,
            reconstructions[row],
            (reconstructions[row] - originals[row]).abs(),
        )

        for column, (image, label) in enumerate(zip(images, labels)):
            axes[row, column].imshow(
                _tensor_to_numpy_image(image)
            )
            axes[row, column].axis("off")

            if row == 0:
                axes[row, column].set_title(label)

    figure.suptitle(title)
    figure.tight_layout()

    figure.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(figure)


def _condition_output_path(
    output_root: Path,
    spec: ModelSpec,
    training_variant: str,
    condition: AttackCondition,
) -> Path:
    return (
        output_root
        / "raw"
        / spec.model_name
        / training_variant
        / f"{condition.file_stem}.csv"
    )


def _preview_output_path(
    output_root: Path,
    spec: ModelSpec,
    training_variant: str,
    condition: AttackCondition,
) -> Path:
    return (
        output_root
        / "previews"
        / spec.model_name
        / training_variant
        / f"{condition.file_stem}.png"
    )


def _write_batch_rows(
    writer: csv.DictWriter,
    *,
    spec: ModelSpec,
    training_variant: str,
    checkpoint_dataset: str,
    checkpoint_path: Path,
    condition: AttackCondition,
    sample_indices: torch.Tensor,
    sample_paths: list[str] | tuple[str, ...],
    class_ids: torch.Tensor,
    input_metrics,
    reconstruction_metrics,
    fidelity_metrics,
) -> None:
    for item in range(sample_indices.shape[0]):
        writer.writerow(
            {
                "model_name": spec.model_name,
                "family": spec.family,
                "latent_channels": spec.latent_channels,
                "latent_elements": spec.latent_elements,
                "training_variant": training_variant,
                "checkpoint_dataset": checkpoint_dataset,
                "evaluation_dataset": "subset",
                "checkpoint_path": str(checkpoint_path),
                "attack": condition.name,
                "severity": condition.severity,
                "direction": condition.direction,
                "attack_space": (
                    "latent" if condition.is_latent else "input"
                ),
                "input_changed": int(condition.input_changed),
                "sample_index": int(sample_indices[item].item()),
                "sample_path": str(sample_paths[item]),
                "class_id": int(class_ids[item].item()),
                "input_mse": float(input_metrics.mse[item].item()),
                "input_psnr": float(input_metrics.psnr[item].item()),
                "input_ssim": float(input_metrics.ssim[item].item()),
                "recon_mse": float(
                    reconstruction_metrics.mse[item].item()
                ),
                "recon_psnr": float(
                    reconstruction_metrics.psnr[item].item()
                ),
                "recon_ssim": float(
                    reconstruction_metrics.ssim[item].item()
                ),
                "fidelity_mse": float(
                    fidelity_metrics.mse[item].item()
                ),
                "fidelity_psnr": float(
                    fidelity_metrics.psnr[item].item()
                ),
                "fidelity_ssim": float(
                    fidelity_metrics.ssim[item].item()
                ),
            }
        )


def evaluate_condition(
    *,
    model: torch.nn.Module,
    spec: ModelSpec,
    training_variant: str,
    checkpoint_dataset: str,
    checkpoint_path: Path,
    dataloader,
    device: torch.device,
    condition: AttackCondition,
    output_root: Path,
    attack_seed: int,
    overwrite: bool,
    save_previews: bool,
    preview_count: int,
) -> Path:
    output_path = _condition_output_path(
        output_root,
        spec,
        training_variant,
        condition,
    )

    if output_path.exists() and not overwrite:
        print(f"[skip] Existing result: {output_path}")
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = output_path.with_suffix(
        output_path.suffix + f".pid{os.getpid()}.tmp"
    )

    if temporary_path.exists():
        temporary_path.unlink()

    preview_saved = False

    try:
        with temporary_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=RAW_COLUMNS,
            )
            writer.writeheader()

            with torch.inference_mode():
                for batch_index, (
                    x,
                    class_ids,
                    sample_indices,
                    sample_paths,
                ) in enumerate(dataloader):
                    x = x.to(
                        device,
                        non_blocking=True,
                    )

                    class_ids = torch.as_tensor(class_ids)
                    sample_indices = torch.as_tensor(sample_indices)

                    attacked_input = apply_input_attack(
                        x,
                        condition,
                        base_seed=attack_seed,
                        batch_index=batch_index,
                    )

                    clean_reconstruction = None

                    if condition.is_latent:
                        reconstruction = _encode_decode_with_latent_attack(
                            model,
                            x,
                            condition,
                            attack_seed=attack_seed,
                            batch_index=batch_index,
                        )

                        if save_previews and not preview_saved:
                            clean_reconstruction = _unwrap_tensor(model(x))
                    else:
                        reconstruction = _unwrap_tensor(
                            model(attacked_input)
                        )

                    reconstruction = torch.clamp(
                        reconstruction,
                        min=0.0,
                        max=1.0,
                    )

                    (
                        input_metrics,
                        reconstruction_metrics,
                        fidelity_metrics,
                    ) = compute_evaluation_metrics(
                        attacked_input=attacked_input,
                        reconstruction=reconstruction,
                        clean_target=x,
                    )

                    _write_batch_rows(
                        writer,
                        spec=spec,
                        training_variant=training_variant,
                        checkpoint_dataset=checkpoint_dataset,
                        checkpoint_path=checkpoint_path,
                        condition=condition,
                        sample_indices=sample_indices,
                        sample_paths=sample_paths,
                        class_ids=class_ids,
                        input_metrics=input_metrics,
                        reconstruction_metrics=reconstruction_metrics,
                        fidelity_metrics=fidelity_metrics,
                    )

                    if save_previews and not preview_saved:
                        direction_text = ""

                        if condition.name == "rotation":
                            signed_angle = (
                                condition.direction * condition.severity
                            )
                            direction_text = f" | angle={signed_angle:+g}"

                        _save_preview(
                            _preview_output_path(
                                output_root,
                                spec,
                                training_variant,
                                condition,
                            ),
                            x,
                            attacked_input,
                            reconstruction,
                            title=(
                                f"{spec.model_name} | "
                                f"{training_variant} | "
                                f"{condition.name} | "
                                f"severity={condition.severity}"
                                f"{direction_text}"
                            ),
                            max_images=preview_count,
                            clean_reconstructions=clean_reconstruction,
                        )

                        preview_saved = True

        os.replace(temporary_path, output_path)

    except torch.cuda.OutOfMemoryError as error:
        if temporary_path.exists():
            temporary_path.unlink()

        torch.cuda.empty_cache()

        raise RuntimeError(
            "CUDA ran out of memory during robustness evaluation. "
            "Restart the command with --batch-size 4 and a new "
            "--output-dir, because batch size is recorded in the run "
            "manifest and must stay constant for all checkpoint results."
        ) from error

    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()

        raise

    print(f"[done] {output_path}")
    return output_path


def evaluate_checkpoint(
    *,
    spec: ModelSpec,
    training_variant: str,
    checkpoint_dataset: str,
    checkpoint_path: Path,
    dataloader,
    conditions: Iterable[AttackCondition],
    device: torch.device,
    output_root: Path,
    attack_seed: int,
    overwrite: bool = False,
    save_previews: bool = False,
    preview_count: int = 4,
) -> list[Path]:
    model = load_trained_model(
        spec,
        checkpoint_path,
        device,
    )

    output_paths: list[Path] = []

    try:
        for condition in conditions:
            output_paths.append(
                evaluate_condition(
                    model=model,
                    spec=spec,
                    training_variant=training_variant,
                    checkpoint_dataset=checkpoint_dataset,
                    checkpoint_path=checkpoint_path,
                    dataloader=dataloader,
                    device=device,
                    condition=condition,
                    output_root=output_root,
                    attack_seed=attack_seed,
                    overwrite=overwrite,
                    save_previews=save_previews,
                    preview_count=preview_count,
                )
            )
    finally:
        del model
        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return output_paths