import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import os
from pathlib import Path
from PIL import Image
from typing import Optional, Literal


class ADNI_Loader(Dataset):
    """ Custom data loader for the ADNI dataset"""

    def __init__(
            self, 
            root: str, 
            split: Literal["test", "train"] = "test", 
            transform: Optional[transforms.Compose] = None
        ):

        """
        root: root folder path of dataset (AD_NC)
        split: dataset split
        transform: any torchvision transforms to be applied
        """
        self.dir = Path(root) / split 
        self.transform = transform

        self.classes = ["AD", "NC"]
        self.label_map = {"AD": 0, "NC": 1}

        self.samples = []

        for class_name in self.classes:
            class_dir = self.dir / class_name
            if not class_dir.exists():
                print(f"Warning: Directory not found: {class_dir}")
                continue

            for img_path in sorted(class_dir.iterdir()):
                self.samples.append((img_path, self.label_map[class_name]))


    def __len__(self) -> int:
        """Returns length of dataset"""
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        """Get item at index"""
        img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {e}")

        if self.transform:
            image = self.transform(image)

        return image, label
    


class Transforms:
    """ Wrapper class holding transforms for the dataset"""

    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.95, 1.0)),
        transforms.RandomAffine(
            degrees=5,
            translate=(0.02, 0.02),
            scale=(0.95, 1.05),
        ),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    aggressive_train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.RandomAffine(
            degrees=10,
            translate=(0.05, 0.05),
            scale=(0.9, 1.1),
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])




if __name__ == "__main__":
    dataset = ADNI_Loader(root="recognition/convnext_47433117/ADNI/AD_NC", split="test", transform=Transforms.test_transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    for images, labels in dataloader:
        print(images.size(), labels)
        break