import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split

full_root = "/raid/kszyc/datasets/ImageNet2012"


def get_transforms(image_size):

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


def get_subnet_datasets(root_dir="datasets/subset_imagenet/", image_size=224, val_split=0.1):

    transform = get_transforms(image_size)
    dataset = ImageFolder(root_dir, transform=transform)

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    return train_set, val_set


def get_imagenet_datasets(root_dir=full_root, image_size=224):

    transform = get_transforms(image_size)

    train_set = ImageFolder(os.path.join(root_dir, "train"), transform=transform)
    val_set = ImageFolder(os.path.join(root_dir, "val"), transform=transform)

    return train_set, val_set