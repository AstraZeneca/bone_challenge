import numpy as np
from src.utils import apply_bone_window, normalize_image, pad_crop_image, load_nii_file
import torch
from torch.utils.data import Dataset


def preprocess_slice(bone_slice, bit_depth, window, size):
    bone_slice = bone_slice.T
    W, L = window

    bone_slice = apply_bone_window(bone_slice, W, L)
    bone_slice = normalize_image(bone_slice, bit_depth=bit_depth)
    bone_slice = pad_crop_image(bone_slice, size)

    return bone_slice


def preprocess(file_path, bit_depth=8, window=(2000, 500), size=(512, 512)):
    img = load_nii_file(file_path)

    center_x, center_y, _ = [dim // 2 for dim in img.shape]

    x_slice = preprocess_slice(img[center_x, :, :], bit_depth, window, size)
    y_slice = preprocess_slice(img[:, center_y, :], bit_depth, window, size)
    combined_slice = normalize_image((0.5 * x_slice + 0.5 * y_slice), bit_depth)

    return np.stack([x_slice, y_slice, combined_slice], axis=-1)


class BoneDatasetTest(Dataset):
    def __init__(self, img_paths, transforms=None):
        self.img_paths = img_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = preprocess(img_path)

        if self.transforms:
            img = self.transforms(image=img)["image"]

        return img


@torch.no_grad
def ensemble_predict(images, models, device="cuda"):
    sum_preds = None

    models = [model.to(device).eval() for model in models]
    images = images.to(device)

    for model in models:
        fold_preds = []

        y_hat = model(images) * images.shape[2]
        fold_preds.append(y_hat.detach().cpu().numpy())

        fold_preds = np.stack(fold_preds)
        print(fold_preds)

        if sum_preds is None:
            sum_preds = fold_preds
        else:
            sum_preds += fold_preds

    avg_preds = np.rint(sum_preds / len(models))

    return avg_preds
