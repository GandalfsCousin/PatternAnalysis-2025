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
        self.patient_ids = []

        for class_name in self.classes:
            class_dir = self.dir / class_name
            if not class_dir.exists():
                print(f"Warning: Directory not found: {class_dir}")
                continue

            for img_path in sorted(class_dir.iterdir()):
                self.samples.append((img_path, self.label_map[class_name]))

                patient_id = img_path.stem.split("_")[0]
                self.patient_ids.append(patient_id)

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
    """Wrapper class holding transforms for ADNI dataset."""

    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
        transforms.RandomAffine(degrees=7, translate=(0.07,0.07), scale=(0.93,1.07)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    aggressive_train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
        transforms.RandomErasing(p=0.5)
    ])

    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])




if __name__ == "__main__":
    dataset = ADNI_Loader(root="recognition/convnext_47433117/ADNI/AD_NC", split="test", transform=Transforms.test_transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    for images, labels in dataloader:
        print(images.size(), labels)
        break