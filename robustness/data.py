from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision.datasets import ImageFolder

from data.data_setter import get_transforms


class IndexedSubset(Dataset):
    """
    Dataset wrapper returning image, class id, source index, and relative path.
    """

    def __init__(
        self,
        dataset: ImageFolder,
        indices: list[int],
        root: Path,
    ) -> None:
        self.dataset = dataset
        self.indices = [int(index) for index in indices]
        self.root = root.resolve()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        source_index = self.indices[item]
        image, target = self.dataset[source_index]

        source_path = Path(
            self.dataset.samples[source_index][0]
        ).resolve()

        try:
            relative_path = source_path.relative_to(self.root)
        except ValueError:
            relative_path = source_path

        return (
            image,
            target,
            source_index,
            str(relative_path),
        )


def _atomic_json_dump(
    payload: dict[str, Any],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(
        path.suffix + f".pid{os.getpid()}.tmp"
    )

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
        )

    os.replace(temporary_path, path)


def _build_subset_validation_split(
    *,
    subset_root: Path,
    image_size: int,
    validation_fraction: float,
    split_seed: int,
) -> tuple[ImageFolder, Subset]:
    if not subset_root.exists():
        raise FileNotFoundError(
            "Subset dataset directory does not exist: "
            f"{subset_root}"
        )

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(
            "validation_fraction must be in (0, 1)."
        )

    dataset = ImageFolder(
        str(subset_root),
        transform=get_transforms(image_size),
    )

    if len(dataset) < 2:
        raise ValueError(
            "The subset dataset must contain at least two images."
        )

    validation_size = max(
        1,
        int(len(dataset) * validation_fraction),
    )
    training_size = len(dataset) - validation_size

    if training_size <= 0:
        raise ValueError(
            "validation_fraction leaves no images outside validation."
        )

    generator = torch.Generator().manual_seed(split_seed)

    _, validation_set = random_split(
        dataset,
        [training_size, validation_size],
        generator=generator,
    )

    return dataset, validation_set


def build_evaluation_loader(
    *,
    subset_root: Path,
    image_size: int,
    validation_fraction: float,
    batch_size: int,
    num_workers: int,
    max_samples: int | None,
    split_seed: int,
    sample_seed: int,
    manifest_path: Path,
    device: torch.device,
    checkpoint_dataset: str,
) -> tuple[DataLoader, dict[str, Any]]:
    """
    Build one deterministic validation subset shared by every checkpoint.

    Evaluation is intentionally restricted to the local ImageNet subset.
    Checkpoint dataset is stored separately because weights are located under
    checkpoints/<model>/full/... while evaluation images come from subset.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0.")

    if num_workers < 0:
        raise ValueError("num_workers must be >= 0.")

    base_dataset, validation_set = _build_subset_validation_split(
        subset_root=subset_root,
        image_size=image_size,
        validation_fraction=validation_fraction,
        split_seed=split_seed,
    )

    requested_samples = (
        len(validation_set)
        if max_samples is None or max_samples <= 0
        else min(int(max_samples), len(validation_set))
    )

    expected_values = {
        "evaluation_dataset": "subset",
        "checkpoint_dataset": checkpoint_dataset,
        "subset_root": str(subset_root.resolve()),
        "image_size": image_size,
        "full_subset_length": len(base_dataset),
        "validation_length": len(validation_set),
        "validation_fraction": validation_fraction,
        "split_seed": split_seed,
        "sample_seed": sample_seed,
        "requested_samples": requested_samples,
        "batch_size": batch_size,
    }

    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)

        for key, expected_value in expected_values.items():
            current_value = manifest.get(key)

            if current_value != expected_value:
                raise ValueError(
                    "Existing run manifest is incompatible for "
                    f"'{key}': found {current_value!r}, "
                    f"expected {expected_value!r}. "
                    "Use a different output directory or remove the old "
                    "run_manifest.json before restarting."
                )

        selected_source_indices = [
            int(value)
            for value in manifest["selected_source_indices"]
        ]
    else:
        generator = torch.Generator().manual_seed(sample_seed)

        selected_validation_positions = (
            torch.randperm(
                len(validation_set),
                generator=generator,
            )[:requested_samples]
            .tolist()
        )

        selected_source_indices = [
            int(validation_set.indices[position])
            for position in selected_validation_positions
        ]

        selected_relative_paths = []

        for source_index in selected_source_indices:
            source_path = Path(
                base_dataset.samples[source_index][0]
            ).resolve()

            try:
                relative_path = source_path.relative_to(
                    subset_root.resolve()
                )
            except ValueError:
                relative_path = source_path

            selected_relative_paths.append(str(relative_path))

        manifest = {
            **expected_values,
            "selected_source_indices": selected_source_indices,
            "selected_relative_paths": selected_relative_paths,
            "normalization": (
                "Resize((image_size, image_size)) + ToTensor(); "
                "image range [0, 1]"
            ),
            "random_attack_note": (
                "Random perturbations are generated from stable seeds per "
                "condition and batch. Batch size is therefore locked by this "
                "manifest for the whole experiment."
            ),
        }

        _atomic_json_dump(manifest, manifest_path)

    indexed_dataset = IndexedSubset(
        dataset=base_dataset,
        indices=selected_source_indices,
        root=subset_root,
    )

    loader_kwargs: dict[str, Any] = {
        "dataset": indexed_dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }

    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    dataloader = DataLoader(**loader_kwargs)

    return dataloader, manifest