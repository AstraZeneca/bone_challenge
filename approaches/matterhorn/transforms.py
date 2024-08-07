import torch
import random
import torchvision.transforms as transforms


class CustomTransform:
    def __init__(self, flip_probability=0.5, rotation_probability=0.5, noise_probability=0.5, intensity_variation_probability=0.5):
        self.flip_probability = flip_probability
        self.rotation_probability = rotation_probability
        self.noise_probability = noise_probability
        self.intensity_variation_probability = intensity_variation_probability

    def add_gaussian_noise(self, image, mean=0, std=0.1):
        noise = torch.randn_like(image) * std + mean
        noisy_image = torch.clamp(image + noise, 0, 1)  # Clamp values to [0, 1]
        return noisy_image

    def apply_intensity_variation(self, image, intensity_factor_range=(0.5, 1.5)):
        intensity_factor = random.uniform(*intensity_factor_range)
        varied_image = torch.clamp(image * intensity_factor, 0, 1)  # Clamp values to [0, 1]
        return varied_image

    def __call__(self, nii_data):
        # Convert numpy array to tensor
        nii_data = torch.tensor(nii_data)

        # Apply random horizontal flip
        if random.random() < self.flip_probability:
            flip = transforms.RandomHorizontalFlip(p=self.flip_probability)
            nii_data = flip(nii_data)

        # Apply random vertical flip
        if random.random() < self.flip_probability:
            flip = transforms.RandomVerticalFlip(p=self.flip_probability)
            nii_data = flip(nii_data)

        # Apply random 90-degree rotation
        if random.random() < self.rotation_probability:
            angle = random.choice([0, 90, 180, 270])
            rotation = transforms.RandomRotation((angle, angle))
            nii_data = rotation(nii_data)

        # Apply Gaussian noise injection
        if random.random() < self.noise_probability:
            nii_data = self.add_gaussian_noise(nii_data, mean=0, std=0.1)

        # Apply random intensity variation
        if random.random() < self.intensity_variation_probability:
            nii_data = self.apply_intensity_variation(nii_data, intensity_factor_range=(0.5, 1.5))

        return nii_data