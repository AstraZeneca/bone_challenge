from torch.utils.data import DataLoader, Dataset
import torch
import lightning.pytorch as pl
from src.utils import LABEL_REGEX, get_all_images
import cv2
import re
import numpy as np
from sklearn.model_selection import GroupKFold
from pathlib import Path


class BoneDataset(Dataset):
    def __init__(self, img_paths, transforms=None):
        self.img_paths = img_paths
        self.transforms = transforms
        self.label_regex = LABEL_REGEX

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if self.transforms:
            img = self.transforms(image=img)["image"]

        label = int(re.search(self.label_regex, img_path).group(1))
        label /= img.squeeze().shape[1]

        return img, torch.tensor(label, dtype=torch.float32)


class BoneDataModule(pl.LightningDataModule):
    def __init__(
        self,
        image_dir: str,
        batch_size: int = 32,
        train_transforms=None,
        val_transforms=None,
        splits=4,
        fold=0,
        num_workers=4,
        random_state=123,
    ):
        super().__init__()
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.num_workers = num_workers
        self.fold = fold
        self.splits = splits
        self.random_state = random_state

    def setup(self, stage):
        all_images = np.array(get_all_images(self.image_dir))
        splits = splitter(all_images, self.splits, self.random_state)
        train_idx, val_idx = splits[self.fold]
        train_images, val_images = all_images[train_idx], all_images[val_idx]

        # Keep only the center slices for validation
        val_images = np.array([path for path in val_images if "x0_y0" in path])

        self.train_dataset = BoneDataset(train_images, self.train_transforms)
        self.val_dataset = BoneDataset(val_images, self.val_transforms)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


# Stratified grouped cross-validation
def splitter(img_paths: list[str], n_splits: int = 3, random_state: int = 0):
    img_paths = sorted(img_paths)

    bone_images = np.array([Path(x).stem.split("_")[0] for x in img_paths])

    print(bone_images)

    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(X=img_paths, groups=bone_images))
