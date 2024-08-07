import os
import random
import numpy as np
from scipy import ndimage
import nibabel as nib
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from tqdm import tqdm
import torch
import gzip
from torch.utils.data import DataLoader

import transforms

def _normalize_Hounsfield_Units(planes):
    # Rescale Hounsfield Units to a reasonable range (-1000 to 3000) for each plane
    clipped_planes = [np.clip(plane, -1000, 3000) for plane in planes]

    # Normalize each plane to the range [0, 1]
    normalized_planes = [
        (plane - (-1000)) / (3000 - (-1000)) for plane in clipped_planes
    ]

    return normalized_planes

def _interpolate_planes(planes, new_size):
    # Interpolate each plane to new_sizexnew_size
    interpolated_planes = [
        ndimage.zoom(
            plane,
            (new_size[0] / plane.shape[0], new_size[1] / plane.shape[1]),
            order=1,
        )
        for plane in planes
    ]
    return interpolated_planes
    
def _add_noise_at_beginning(array, width_to_add, duplicate_col=2):
    height, width = array.shape

    # Extract the content to duplicate
    content_to_add = array[:, duplicate_col].reshape(-1, 1)

    # Add random width to the beginning of each row
    modified_array = np.zeros((height, width + width_to_add), dtype=array.dtype)
    modified_array[:, width_to_add : width_to_add + width] = array

    # Copy the content to duplicate to the beginning of each row
    for i in range(width_to_add):
        # modified_array[:, i] = content_to_add.flatten()  # Flatten to match the dimension
        # Random shuffle of content_to_add
        shuffled_content = np.random.permutation(content_to_add.flatten())
        modified_array[:, i] = shuffled_content

    # Remove the same number of width elements from the end of each row
    modified_array = modified_array[:, :width]

    return modified_array

def _add_noise_at_end(array, width_to_add, duplicate_col=-2):
    height, width = array.shape

    # Extract the content to duplicate
    content_to_add = array[:, duplicate_col].reshape(-1, 1)

    # Add random width to the end of each row
    modified_array = np.zeros((height, width + width_to_add), dtype=array.dtype)
    modified_array[:, :width] = array

    # Add randomized content at the end of each row
    for i in range(width, width + width_to_add):
        shuffled_content = np.random.permutation(content_to_add.flatten())
        modified_array[:, i] = shuffled_content

    # Remove the same number of width elements from the beginning of each row
    modified_array = modified_array[:, width_to_add:]

    return modified_array

class AxialDataset(Dataset):
    def __init__(
        self,
        file_paths,
        labels=None,
        transform=None,
        img_sz=[300, 642],
        central_prop=0.5,
        axis=[0, 1],
        growth_plane_position_augmentation=False,
        zaxis_mode="crop",
        planes_index_selection="consecutive",
        num_input_channels = 3,
    ):
        self.file_paths = file_paths
        self.image_names = [os.path.splitext(os.path.basename(path))[0] for path in file_paths]
        self.labels = labels
        self.transform = transform
        self.img_sz = img_sz
        self.central_proportion = central_prop
        self.axis = axis
        self.growth_plane_position_augmentation = growth_plane_position_augmentation
        self.zaxis_mode = zaxis_mode
        self.planes_index_selection = planes_index_selection
        self.num_input_channels = num_input_channels

    def __len__(self):
        return len(self.file_paths)

    def _get_random_uniform_planes(self, nii_data):
        # Randomly choose x or y axis
        axis = random.choice(self.axis)

        # Get 'self.input_channels' random planes along the chosen axis
        if axis == 0:
            indices = random.sample(range(nii_data.shape[0]), self.num_input_channels)
            indices.sort() # random but sorted
            planes = [nii_data[i, :, :] for i in indices]
        else:
            indices = random.sample(range(nii_data.shape[1]), self.num_input_channels)
            indices.sort() # random but sorted
            planes = [nii_data[:, i, :] for i in indices]

        return planes

    def _get_random_centered_uniform_planes(self, nii_data, label):
        axis = random.choice(self.axis)
        new_label = label  # no change if not growth_plane_position_augmentation

        # Calculate the number of central indices
        dim = nii_data.shape[0] if axis == 0 else nii_data.shape[1]
        num_central_indices = int(dim * self.central_proportion)
        central_indices_range = range(
            (dim - num_central_indices) // 2,
            (dim + num_central_indices) // 2,
        )
        if self.planes_index_selection == "random":
            indices = random.sample(central_indices_range, self.num_input_channels)
            indices.sort() # random but sorted
        else:
            # planes_index_selection consecutive
            index = random.choice(central_indices_range)
            indices = [index - self.num_input_channels//2 + i for i in range(self.num_input_channels)]

        # Get the planes corresponding to the selected indices
        planes = [
            nii_data[
                (i if axis == 0 else slice(None)), (slice(None) if axis == 0 else i), :
            ]
            for i in indices
        ]

        if self.growth_plane_position_augmentation:
            new_label = np.random.randint(0, planes[0].shape[1])
            if self.do == "crop":
                new_label = np.random.randint(0, self.img_sz[0])
            if new_label < label:
                for i in range(self.num_input_channels):
                    planes[i] = _add_noise_at_end(
                        planes[i], width_to_add=int(label - new_label)
                    )
            elif new_label > label:
                for i in range(self.num_input_channels):
                    planes[i] = _add_noise_at_beginning(
                        planes[i], width_to_add=int(new_label - label)
                    )

        return planes, new_label

    def _calculate_do_zaxis(self):
        if self.zaxis_mode == "crop":
            self.do = "crop"
        elif self.zaxis_mode == "interpolate":
            self.do = "interpolate"
        elif self.zaxis_mode == "mixed":
            self.do = random.choice(["crop", "interpolate"])
        else:
            assert 1==0, "unknown zaxis mode"        

    def _calculate_size_and_label(self, label):            
        if self.do == "crop":
            interpolate_sz = [self.img_sz[0], self.img_sz[1]]
            # label the same
            #print(f"previous label assert {label}")
            assert label <= self.img_sz[0]
        elif self.do == "interpolate":
            interpolate_sz = [self.img_sz[0], self.img_sz[0]]
            label = int(label / float(self.img_sz[1]) * float(self.img_sz[0]))  
              
        return interpolate_sz, label    

    def __getitem__(self, idx):
        # calculate if "crop" or "interpolate" for this item and save in self.do
        self._calculate_do_zaxis()
        
        nii_file_path = self.file_paths[idx]
        nii_data = nib.load(nii_file_path).get_fdata().astype(np.float32)

        if self.labels is not None:
            label = self.labels[idx]  # normalized at the end of the function
        else:
            label = None
        # if growth_plane_position_augmentation on then the next function can modify label (plane)
        planes, label = self._get_random_centered_uniform_planes(nii_data, label=label)

        # Normalize each plane to the range [0, 1]
        normalized_planes = _normalize_Hounsfield_Units(planes)

        interpolate_sz, label = self._calculate_size_and_label(label)
        
        # Resize by interpolation
        interpolated_planes = _interpolate_planes(normalized_planes, new_size=interpolate_sz)

        # Stack the planes to form a set of channels image
        stacked_image = np.stack(interpolated_planes)

        # after interpolation, crop the begining of the image (if crop), if not (interpolate) it does nothing
        stacked_image = stacked_image[:, : self.img_sz[0], : self.img_sz[0]]

        if self.transform:
            stacked_image = self.transform(stacked_image)

        if self.labels is not None:
            return stacked_image, label
        else:
            return stacked_image

def save_compressed_tensor(file_path, tensor, label):
    with gzip.open(file_path, 'wb') as f:
        # Save both the tensor and label in a dictionary
        torch.save({'image': tensor, 'label': label}, f)

def load_compressed_tensor(file_path):
    with gzip.open(file_path, 'rb') as f:
        data = torch.load(f, weights_only=True)
    return data['image'], data['label']

class PrecomputedAxialDataset(Dataset):
    def __init__(
        self,
        original_dataset,
        augmentation_count,
        augmentations_save_dir,
        use_existing_augmented_dir=False
    ):
        self.original_dataset = original_dataset
        self.augmentation_count = augmentation_count
        self.augmentations_save_dir = augmentations_save_dir
        self.use_existing_augmented_dir = use_existing_augmented_dir

        if not os.path.exists(self.augmentations_save_dir):
            os.makedirs(self.augmentations_save_dir)

        self.augmented_files = self._generate_or_load_augmented_files()

    def __len__(self):
        return len(self.original_dataset)

    def _generate_or_load_augmented_files(self):
        augmented_files = {}

        if not self.use_existing_augmented_dir:
            for idx in tqdm(range(len(self.original_dataset)), desc="Generating Augmented Files"):
                augmented_files[idx] = []
                for aug_idx in range(self.augmentation_count):
                    compressed_file_path = os.path.join(
                        self.augmentations_save_dir,
                        f"aug_{idx}_{aug_idx}.pt.gz"
                    )
                    
                    if not os.path.exists(compressed_file_path):
                        image, new_label = self.original_dataset.__getitem__(idx)
                        save_compressed_tensor(compressed_file_path, image, new_label)

                    augmented_files[idx].append(compressed_file_path)
        else:
            existing_files = [f for f in os.listdir(self.augmentations_save_dir) if f.endswith('.pt.gz')]
            if not existing_files:
                raise FileNotFoundError("No augmented tensor files found in the specified directory.")

            for file_name in existing_files:
                parts = file_name.split('_')
                idx = int(parts[1])
                aug_idx_with_extension = parts[2].split('.')[0]
                aug_idx = int(aug_idx_with_extension)
                if idx not in augmented_files:
                    augmented_files[idx] = []
                augmented_files[idx].append(os.path.join(self.augmentations_save_dir, file_name))

        return augmented_files

    def __getitem__(self, idx):
        file_path = random.choice(self.augmented_files[idx])
        image, label = load_compressed_tensor(file_path)
        return image, label
        
def create_train_dataloaders(
    fold,
    train_file_paths,
    val_file_paths, 
    train_labels,
    val_labels, 
    img_sz, 
    central_prop, 
    central_prop_val, 
    zaxis_mode, 
    planes_index_selection, 
    num_input_channels,
    augmented_dataset_root_dir,
    use_existing_augmented_dir,   
    augmentation_train_count,
    augmentation_valid_count,
    batch_size,
    num_workers    
):
    transformations = Compose([
        transforms.CustomTransform(
            flip_probability=0.5, 
            rotation_probability=0.5, 
            noise_probability=0.5, 
            intensity_variation_probability=0.5
        )
    ])
    
    val_transformations = Compose([
        transforms.CustomTransform(
            flip_probability=0.0, 
            rotation_probability=0.0, 
            noise_probability=0.0, 
            intensity_variation_probability=0.0
        )
    ])  
        
    train_dataset = AxialDataset(
        file_paths=train_file_paths,
        labels=train_labels,
        transform=transformations,
        img_sz=img_sz,
        central_prop=central_prop,
        axis=[0, 1],
        growth_plane_position_augmentation=True,
        zaxis_mode=zaxis_mode,
        planes_index_selection=planes_index_selection,
        num_input_channels=num_input_channels,
    )
    
    precomputed_train_dataset = PrecomputedAxialDataset(
        original_dataset = train_dataset,
        augmentation_count = augmentation_train_count,
        augmentations_save_dir = f"{augmented_dataset_root_dir}/train/fold_{fold}",
        use_existing_augmented_dir=use_existing_augmented_dir
    )

    val_dataset_xy = AxialDataset(
        file_paths=val_file_paths,
        labels=val_labels,
        transform=val_transformations,
        img_sz=img_sz,
        central_prop=central_prop_val,
        axis=[0, 1],
        growth_plane_position_augmentation=False,
        zaxis_mode="crop",
        planes_index_selection=planes_index_selection,
        num_input_channels=num_input_channels,
    )
    
    precomputed_val_dataset = PrecomputedAxialDataset(
        original_dataset = val_dataset_xy,
        augmentation_count = augmentation_valid_count,
        augmentations_save_dir = f"{augmented_dataset_root_dir}/valid/fold_{fold}",
        use_existing_augmented_dir=use_existing_augmented_dir
    )    

    # Create data loaders for training and validation
    train_data_loader = DataLoader(
        precomputed_train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    val_data_loader_xy = DataLoader(
        precomputed_val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )

    return train_data_loader, val_data_loader_xy


def create_eval_dataloaders(
    val_file_paths, 
    val_labels, 
    img_sz, 
    central_prop_val, 
    planes_index_selection, 
    num_input_channels,
    batch_size,
    num_workers       
):
    val_transformations = Compose([
        transforms.CustomTransform(
            flip_probability=0.0, 
            rotation_probability=0.0, 
            noise_probability=0.0, 
            intensity_variation_probability=0.0
        )
    ])  
        
    val_dataset_x = AxialDataset(
        file_paths=val_file_paths,
        labels=val_labels,
        transform=val_transformations,
        img_sz=img_sz,
        central_prop=central_prop_val,
        axis=[0],
        growth_plane_position_augmentation=False,
        zaxis_mode="crop",
        planes_index_selection=planes_index_selection,
        num_input_channels=num_input_channels,
    )

    val_dataset_y = AxialDataset(
        file_paths=val_file_paths,
        labels=val_labels,
        transform=val_transformations,
        img_sz=img_sz,
        central_prop=central_prop_val,
        axis=[1],
        growth_plane_position_augmentation=False,
        zaxis_mode="crop",
        planes_index_selection=planes_index_selection,
        num_input_channels=num_input_channels,
    )
    
    dataloader_x = DataLoader(val_dataset_x, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    dataloader_y = DataLoader(val_dataset_y, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    return dataloader_x, dataloader_y

