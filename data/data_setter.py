import os
import json
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, Dataset


class ImageNetKaggle(Dataset):
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


def get_transforms(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()  # normalizacja do [0,1]
    ])


def get_subnet_datasets(root_dir="datasets/subset_imagenet/", image_size=224, val_split=0.1):
    transform = get_transforms(image_size)
    dataset = ImageFolder(root_dir, transform=transform)

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    return train_set, val_set


def get_imagenet_datasets(root_dir="/raid/kszyc/datasets/ImageNet2012", image_size=224):
    transform = get_transforms(image_size)

    train_set = ImageNetKaggle(root=root_dir, split="train", transform=transform)
    val_set = ImageNetKaggle(root=root_dir, split="val", transform=transform)

    return train_set, val_set