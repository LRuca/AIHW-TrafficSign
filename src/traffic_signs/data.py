import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def build_transforms(image_size, policy, is_train):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if not is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])

    ops = [transforms.Resize((image_size, image_size))]

    if policy == "basic":
        ops.extend([
            transforms.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        ])
    elif policy == "basic_color":
        ops.extend([
            transforms.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
        ])
    elif policy == "basic_geometric":
        ops.extend([
            transforms.RandomAffine(degrees=12, translate=(0.08, 0.08), scale=(0.9, 1.1), shear=6),
        ])
    elif policy == "none":
        pass
    else:
        raise ValueError("Unsupported augmentation policy: {}".format(policy))

    ops.extend([transforms.ToTensor(), normalize])
    return transforms.Compose(ops)


def scan_imagefolder(root_dir):
    class_names = sorted(
        [name for name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, name))],
        key=lambda item: int(item) if item.isdigit() else item,
    )
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    samples = []
    for class_name in class_names:
        class_dir = os.path.join(root_dir, class_name)
        for file_name in sorted(os.listdir(class_dir)):
            if file_name.lower().endswith(IMAGE_EXTENSIONS):
                samples.append((os.path.join(class_dir, file_name), class_to_idx[class_name]))
    return samples, class_names, class_to_idx


def split_samples(samples, train_ratio, seed):
    label_to_items = {}
    for path, label in samples:
        label_to_items.setdefault(label, []).append((path, label))

    rng = random.Random(seed)
    train_samples = []
    val_samples = []
    for label, items in label_to_items.items():
        rng.shuffle(items)
        split_index = max(1, int(len(items) * train_ratio))
        split_index = min(split_index, len(items) - 1)
        train_samples.extend(items[:split_index])
        val_samples.extend(items[split_index:])
    return train_samples, val_samples


class TrafficSignDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        image = self.transform(image)
        return image, label, image_path


def compute_class_weights(samples, num_classes):
    counts = np.zeros(num_classes, dtype=np.float32)
    for _, label in samples:
        counts[label] += 1.0
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def build_dataloaders(config):
    data_cfg = config["data"]
    train_root = data_cfg["train_dir"]
    val_root = data_cfg["val_dir"]
    test_root = data_cfg.get("test_dir")
    num_workers = int(data_cfg["num_workers"])
    if os.environ.get("TRAFFIC_SIGNS_FORCE_NUM_WORKERS_ZERO") == "1":
        num_workers = 0

    all_train_samples, class_names, class_to_idx = scan_imagefolder(train_root)
    if data_cfg.get("use_existing_val_dir", True) and val_root and os.path.isdir(val_root):
        val_samples, _, _ = scan_imagefolder(val_root)
        train_samples = all_train_samples
    else:
        train_samples, val_samples = split_samples(all_train_samples, data_cfg["train_split"], config["train"]["seed"])

    image_size = data_cfg["image_size"]
    train_transform = build_transforms(image_size, config["augmentation"]["policy"], is_train=True)
    eval_transform = build_transforms(image_size, config["augmentation"]["policy"], is_train=False)

    train_dataset = TrafficSignDataset(train_samples, train_transform)
    val_dataset = TrafficSignDataset(val_samples, eval_transform)
    test_samples = []
    test_dataset = None
    if test_root and os.path.isdir(test_root):
        test_samples, _, _ = scan_imagefolder(test_root)
        test_dataset = TrafficSignDataset(test_samples, eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config["train"]["batch_size"],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "test_samples": test_samples,
        "class_names": class_names,
        "class_to_idx": class_to_idx,
    }
