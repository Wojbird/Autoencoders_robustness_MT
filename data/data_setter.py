import os
import random
import json
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, Dataset


class ImageNetKaggle(Dataset):
    """
    Custom Dataset for loading ImageNet2012 in Kaggle-style format with ILSVRC2012_val_labels.json
    and imagenet_class_index.json.
    """

    def __init__(self, root, split, transform=None):
        self.samples = []
        self.targets = []
        self.transform = transform
        self.syn_to_class = {}

        with open(os.path.join(root, "imagenet_class_index.json"), "rb") as f:
            json_file = json.load(f)
            for class_id, v in json_file.items():
                self.syn_to_class[v[0]] = int(class_id)

        with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "rb") as f:
            self.val_to_syn = json.load(f)

        samples_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC", split)

        if split == "train":
            for syn_id in sorted(os.listdir(samples_dir)):
                syn_folder = os.path.join(samples_dir, syn_id)
                if os.path.isdir(syn_folder):
                    target = self.syn_to_class[syn_id]
                    for sample in sorted(os.listdir(syn_folder)):
                        sample_path = os.path.join(syn_folder, sample)
                        self.samples.append(sample_path)
                        self.targets.append(target)
        elif split == "val":
            for entry in sorted(os.listdir(samples_dir)):
                syn_id = self.val_to_syn[entry]
                target = self.syn_to_class[syn_id]
                sample_path = os.path.join(samples_dir, entry)
                self.samples.append(sample_path)
                self.targets.append(target)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            x = self.transform(x)
        return x, self.targets[idx]


def get_subnet_dataloaders(
    data_dir: str,
    image_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 8,
    seed: int = 42
):
    """
    Returns train/val dataloaders for the Subset of ImageNet (ImageFolder).
    Uses a reproducible random split.
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),  # normalized to [0,1]
    ])

    dataset = ImageFolder(data_dir, transform=transform)

    total = len(dataset)
    train_size = int(0.85 * total)
    val_size = total - train_size

    random.seed(seed)
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


def get_imagenet_dataloaders(
    root: str,
    image_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 4
):
    """
    Returns train and val dataloaders for full ImageNet2012.
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),  # normalized to [0,1]
    ])

    train_set = ImageNetKaggle(root, split="train", transform=transform)
    val_set = ImageNetKaggle(root, split="val", transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader