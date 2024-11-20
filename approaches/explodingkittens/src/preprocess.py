import pandas as pd
import numpy as np
import cv2
from joblib import Parallel, delayed
import os
from tqdm import tqdm

from src.utils import (
    apply_bone_window,
    equalize_hist_16bit,
    normalize_image,
    pad_crop_image,
    load_nii_file,
)


class BoneProcessor:
    def __init__(
        self,
        metadata: pd.DataFrame,
        output_dir: str = "2_5D_images",
        size: tuple[int] = (512, 512),
        bit_depth: int = 16,
        num_jobs: int = 1,
        offset_step: int = 5,
        apply_histogram_equalization: bool = False,
        window: tuple[int] = None,
    ):
        self.file_info = zip(
            metadata["image_path"].values,
            metadata["Image Name"].values,
            metadata["Growth Plate Index"].values,
        )
        self.output_dir = output_dir
        self.size = size
        self.bit_depth = bit_depth
        self.offset_step = offset_step
        self.num_jobs = num_jobs
        self.apply_histogram_equalization = apply_histogram_equalization
        self.window = window

    def process_scan(self, image: np.ndarray):
        processed_scans = []
        metadata = []
        center_x, center_y, _ = [dim // 2 for dim in image.shape]

        start_slice = -3 * self.offset_step
        end_slice = 3 * self.offset_step + 1

        for offset_x in range(start_slice, end_slice, self.offset_step):
            new_x = center_x + offset_x
            if 0 <= new_x < image.shape[0]:
                x_slice = image[new_x, :, :]
                x_slice = self.process_slice(x_slice)

                for offset_y in range(start_slice, end_slice, self.offset_step):
                    new_y = center_y + offset_y
                    if 0 <= new_y < image.shape[1]:
                        y_slice = image[:, new_y, :]
                        y_slice = self.process_slice(y_slice)

                        combined_slice = normalize_image(
                            (0.5 * x_slice + 0.5 * y_slice), self.bit_depth
                        )
                        processed_scans.append(
                            np.stack([x_slice, y_slice, combined_slice], axis=-1)
                        )
                        metadata.append((offset_x, offset_y))

        return (processed_scans, metadata)

    def process_files(self):
        os.makedirs(self.output_dir, exist_ok=True)

        Parallel(n_jobs=self.num_jobs)(
            delayed(self.process_and_save_images)(file_path, file_idx, gp_idx)
            for file_path, file_idx, gp_idx in tqdm(self.file_info)
        )

        # for file_path, file_idx, gp_idx in self.file_info:
        #    self.process_and_save_images(file_path, file_idx, gp_idx)

    def process_and_save_images(self, file_path, file_idx, gp_idx):
        image = load_nii_file(file_path)
        processed_slices, metadata = self.process_scan(image)

        for i, slice_image in enumerate(processed_slices):
            pos_x, pos_y = metadata[i]
            cv2.imwrite(
                f"{self.output_dir}/{file_idx}_slice_x{pos_x}_y{pos_y}_gp_{gp_idx}.png",
                slice_image,
            )

    def process_slice(self, img_slice):
        # Apply windowing and histogram equalization if necessary
        img_slice = img_slice.T
        if self.window:
            W, L = self.window
            img_slice = apply_bone_window(img_slice, W, L)

        img_slice = normalize_image(img_slice, bit_depth=self.bit_depth)

        if self.apply_histogram_equalization:
            img_slice = equalize_hist_16bit(img_slice, W, L)

        img_slice = pad_crop_image(img_slice, self.size)

        return img_slice
