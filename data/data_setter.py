import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from utils.path_loader import load_imagenet_root


def get_transforms(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])


def get_subnet_datasets(root_dir="datasets/subset_imagenet/", image_size=224, val_split=0.1):
    transform = get_transforms(image_size)
    dataset = ImageFolder(root=root_dir, transform=transform)

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    return train_set, val_set


def get_imagenet_datasets(root_dir=None, image_size=224):
    if root_dir is None:
        root_dir = load_imagenet_root()

    transform = get_transforms(image_size)

    train_dir = os.path.join(root_dir, "train")
    val_dir = os.path.join(root_dir, "val")

    train_set = ImageFolder(root=train_dir, transform=transform)
    val_set = ImageFolder(root=val_dir, transform=transform)

    return train_set, val_set