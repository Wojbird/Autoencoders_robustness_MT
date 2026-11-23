import json
import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder


def _looks_like_imagenet_root(path: Path) -> bool:
    return (
        (path / "imagenet_class_index.json").is_file()
        and (path / "ILSVRC2012_val_labels.json").is_file()
        and (path / "ILSVRC" / "Data" / "CLS-LOC").is_dir()
    )


def load_dataset_root(
    config_path: str = "data/root.json",
) -> str:
    environment_value = os.environ.get("IMAGENET_DIR")

    if environment_value:
        environment_path = Path(environment_value).expanduser()

        if _looks_like_imagenet_root(environment_path):
            return str(environment_path)

        nested_path = environment_path / "ImageNet2012"
        if _looks_like_imagenet_root(nested_path):
            return str(nested_path)

        raise FileNotFoundError(
            "IMAGENET_DIR does not point to an ImageNet2012 directory "
            "or its parent: "
            f"{environment_path}"
        )

    config_file = Path(config_path)
    if config_file.is_file():
        with config_file.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        if isinstance(payload, str):
            configured_path = Path(payload).expanduser()
        elif isinstance(payload, dict):
            value = (
                payload.get("imagenet_root")
                or payload.get("dataset_root")
                or payload.get("root")
            )
            if not value:
                raise ValueError(
                    f"{config_file} must define 'imagenet_root', "
                    "'dataset_root', or 'root'."
                )
            configured_path = Path(value).expanduser()
        else:
            raise ValueError(
                f"Unsupported JSON structure in {config_file}."
            )

        if _looks_like_imagenet_root(configured_path):
            return str(configured_path)

        nested_path = configured_path / "ImageNet2012"
        if _looks_like_imagenet_root(nested_path):
            return str(nested_path)

        raise FileNotFoundError(
            f"Configured ImageNet path is invalid: {configured_path}"
        )

    fallback_path = Path("/data/ImageNet2012")
    if _looks_like_imagenet_root(fallback_path):
        return str(fallback_path)

    raise FileNotFoundError(
        "ImageNet2012 root was not found. Set IMAGENET_DIR or create "
        f"{config_file}."
    )


class ImageNetKaggle(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        transform=None,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'.")

        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.samples: list[str] = []
        self.targets: list[int] = []

        class_index_path = self.root / "imagenet_class_index.json"
        validation_labels_path = (
            self.root / "ILSVRC2012_val_labels.json"
        )

        with class_index_path.open("r", encoding="utf-8") as file:
            class_index = json.load(file)

        synset_to_class = {
            value[0]: int(class_id)
            for class_id, value in class_index.items()
        }

        samples_dir = (
            self.root
            / "ILSVRC"
            / "Data"
            / "CLS-LOC"
            / split
        )

        if not samples_dir.is_dir():
            raise FileNotFoundError(
                f"ImageNet split directory does not exist: {samples_dir}"
            )

        if split == "train":
            for synset_id in sorted(os.listdir(samples_dir)):
                synset_directory = samples_dir / synset_id

                if not synset_directory.is_dir():
                    continue
                if synset_id not in synset_to_class:
                    raise KeyError(
                        f"Unknown ImageNet synset directory: {synset_id}"
                    )

                target = synset_to_class[synset_id]

                for filename in sorted(os.listdir(synset_directory)):
                    sample_path = synset_directory / filename
                    if sample_path.is_file():
                        self.samples.append(str(sample_path))
                        self.targets.append(target)

        else:
            with validation_labels_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                validation_to_synset = json.load(file)

            for filename in sorted(os.listdir(samples_dir)):
                sample_path = samples_dir / filename
                if not sample_path.is_file():
                    continue

                if filename not in validation_to_synset:
                    raise KeyError(
                        "Missing validation label for ImageNet file: "
                        f"{filename}"
                    )

                synset_id = validation_to_synset[filename]
                target = synset_to_class[synset_id]

                self.samples.append(str(sample_path))
                self.targets.append(target)

        if not self.samples:
            raise ValueError(
                f"No images were found in ImageNet split: {samples_dir}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample_path = self.samples[index]

        with Image.open(sample_path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, self.targets[index]


def get_transforms(image_size: int):
    image_size = int(image_size)

    if image_size <= 0:
        raise ValueError("image_size must be positive.")

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


def get_subnet_datasets(
    root_dir: str | Path = "datasets/subset_imagenet/",
    image_size: int = 224,
    val_split: float = 0.1,
    split_seed: int = 42,
):
    val_split = float(val_split)

    if not 0.0 < val_split < 1.0:
        raise ValueError("val_split must satisfy 0.0 < val_split < 1.0.")

    root_path = Path(root_dir)
    if not root_path.is_dir():
        raise FileNotFoundError(
            f"Subset dataset directory does not exist: {root_path}"
        )

    dataset = ImageFolder(
        str(root_path),
        transform=get_transforms(image_size),
    )

    if len(dataset) < 2:
        raise ValueError(
            "The subset dataset must contain at least two images."
        )

    validation_size = max(1, int(len(dataset) * val_split))
    training_size = len(dataset) - validation_size

    if training_size <= 0:
        raise ValueError("val_split leaves no images for training.")

    generator = torch.Generator().manual_seed(int(split_seed))

    return random_split(
        dataset,
        [training_size, validation_size],
        generator=generator,
    )


def get_imagenet_datasets(
    root_dir: str | Path | None = None,
    image_size: int = 224,
):
    if root_dir is None:
        root_dir = load_dataset_root()

    transform = get_transforms(image_size)

    train_set = ImageNetKaggle(
        root_dir,
        "train",
        transform=transform,
    )
    validation_set = ImageNetKaggle(
        root_dir,
        "val",
        transform=transform,
    )

    return train_set, validation_set