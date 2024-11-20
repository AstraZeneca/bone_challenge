import torch
from torch.utils.data import Dataset
from numpy.typing import ArrayLike
from typing import Union, Tuple, List
import pandas as pd
import numpy as np
import json
import os
from torchvision.transforms import v2
from PIL import Image


class BoneSlicesDataset(Dataset):
    """
    A PyTorch Dataset for bone slice images with metadata.

    Args:
        json_config_filepath (str): Path to the config JSON file.
        transform (callable, optional): Transformation to apply to images.
        permute_order (List[int], optional): Order to permute image dimensions.
    """

    def __init__(self, json_config_filepath: str, transform = None, permute_order: List = [2,0,1]) -> None:
        """
        Initializes the dataset with config and metadata.

        Args:
            json_config_filepath (str): Path to the config JSON file.
            transform (callable, optional): Transformation to apply to images.
            permute_order (List[int], optional): Order to permute image dimensions.
        """
        with open(json_config_filepath) as json_file:
            self.config = json.load(json_file)
        self.metadata = self.get_metadata()
        self.metadata = self.metadata.reset_index()
        self.permute_order = permute_order
        if transform is None:
             self.transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=False)])
        else:
            self.transform = transform

    def __len__(self):
        """
        Returns:
            int: Number of samples in dataset.
        """
        return self.metadata.shape[0]

    def __getitem__(self, idx: Union[int, ArrayLike]) -> Tuple[torch.tensor, float]:
        """
        Gets image and label for a given index.
        Args:
            idx (Union[int, ArrayLike]): Index of the sample.
        Returns:
            Tuple[torch.tensor, float]: Image tensor and label.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = self.metadata.loc[idx, 'img_path']
        slice_idx = self.metadata.loc[idx, self.config['slice_index_col']]
        growth_plate_idx = self.metadata.loc[idx, self.config['growth_plate_index_col']]
        img = torch.tensor(np.load(image_path))
        img = img.permute(self.permute_order[0], self.permute_order[1], self.permute_order[2])
        label = self.get_label(self.config['label_option'], slice_idx, growth_plate_idx)
        img = self.transform(img)
        return img, label
    
    def get_metadata(self) -> pd.DataFrame:
        """
        Returns:
            pd.DataFrame: Metadata DataFrame or None.
        """
        df =  pd.read_csv(self.config['metadata_with_slices_path'])
        df['img_path'] = [os.path.join(self.config['slices_data_dir'], img_name) for img_name in df['img_file_name']]
        return df

    def get_label(self, option: str, slice_idx: int, growth_plate_idx: int):
        """
        Depending on the problem we want to solve we can provide different types of labels in option parameter
        Possible options: growth_plate_index|distance_to_growth_plate|absolute_distance_to_growth_plate|is_before_growth_plate
        Args:
            option (str): Label option.
            slice_idx (int): Slice index.
            growth_plate_idx (int): Growth plate index.

        Returns:
            float: Computed label.

        Raises:
            Exception: If the option is invalid.
        """
        if option == 'growth_plate_index':
            return growth_plate_idx

        elif option == 'distance_to_growth_plate':
            return growth_plate_idx - slice_idx

        elif option == 'absolute_distance_to_growth_plate':
            return np.abs(growth_plate_idx - slice_idx)

        elif option == 'is_before_growth_plate':
            return int(slice_idx < growth_plate_idx)

        else:
            raise Exception("Invalid value for label_option in config.json")


    def subset_by_study_id_and_bone_id(self, subset_list: List[Tuple[int, int]]) -> None:
        """
        Filters dataset by (study_id, bone_id) tuples.

        Args:
            subset_list (List[Tuple[int, int]]): List of (study_id, bone_id) tuples.
        """
        result = []
        for study_id, bone_id in subset_list:
            study_id_mask = self.metadata['STUDY ID'].isin([study_id])
            bone_id_mask = self.metadata['Bone ID'].isin([bone_id])
            result.append(self.metadata.loc[study_id_mask & bone_id_mask])
        self.metadata = pd.concat(result)
        self.metadata.reset_index(inplace=True)


    def subset_by_study_id(self, study_id: List[int]) -> None:
        """
        Filters dataset by study IDs.

        Args:
            study_id (List[int]): List of study IDs.
        """
        self.metadata = self.metadata.loc[self.metadata['STUDY ID'].isin(study_id)]
        self.metadata.reset_index(inplace=True)

    def subset_by_bone_id(self, bone_id: List[int]) -> None:
        """
        Filters dataset by bone IDs.

        Args:
            bone_id (List[int]): List of bone IDs.
        """
        self.metadata = self.metadata.loc[self.metadata['Bone ID'].isin(bone_id)]
        self.metadata.reset_index(inplace=True)
        
    def subset_by_image_name(self, img_name: List[str]) -> None:
        """
        Filters dataset by image names.

        Args:
            img_name (List[str]): List of image names.
        """
        self.metadata = self.metadata[self.metadata['Image Name'].isin(img_name)]
        self.metadata.reset_index(inplace = True)
    
    def subset_slices(self, fraction: float, axis: str):
        """
        Subsets slices within a fraction of the center.

        Args:
            fraction (float): Fraction of slices to keep.
            axis (str): Axis to filter ('x', 'y', or 'z').
        """
        center = (self.metadata['Slice Index'].max() - self.metadata['Slice Index'].min()) //2
        offset = fraction * (self.metadata['Slice Index'].max() - self.metadata['Slice Index'].min()) // 2
        lower_bound, upper_bound = center - offset, center + offset
        axis_mask = self.metadata['axis'] == axis
        lower_bound_mask = self.metadata['Slice Index'] > lower_bound
        upper_bound_mask = self.metadata['Slice Index'] < upper_bound
        self.metadata = self.metadata[axis_mask & lower_bound_mask & upper_bound_mask]
        self.metadata.drop('level_0', axis=1, inplace = True)
        self.metadata.reset_index(inplace = True)
        
    def oversample_minority_class(self):
        """
        Creates a mask for the minority class.

        Returns:
            pd.Series: Mask indicating the minority class.
        """
        is_before_growth_plate = self.metadata['Slice Index'] < self.metadata['Growth Plate Index']
        return is_before_growth_plate