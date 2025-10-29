import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules import small_model
from dataset import ADNI_Loader, Transforms

 

class Config:
    dataRoot = "recognition/convnext_47433117/ADNI/AD_NC"
    batchSize = 16
    epochs = 50
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    learningRate = 1e-3


def main():
    train_dataset = ADNI_Loader(Config.dataRoot, split="train", transform=Transforms.train_transform)
    test_dataset = ADNI_Loader(Config.dataRoot, split="test", transform=Transforms.test_transform)
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=Config.batchSize, shuffle=True, pin_memory=True, num_workers=6)
    test_loader = DataLoader(test_dataset, batch_size=Config.batchSize, shuffle=False, pin_memory=True, num_workers=6)
    

    model = small_model().to(Config.device)
    criterion = nn.CrossEntropyLoss()
    total_step = len(train_loader)
    optimiser = torch.optim.Adam(model.parameters(), lr=Config.learningRate)

    model.train()

    for epoch in range(Config.epochs):
        loop = tqdm(train_loader, desc="Training", leave=False)
        for images, labels in loop:
            images, labels = images.to(Config.device), labels.to(Config.device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        print(loss.item()) # type: ignore


if __name__ == "__main__":
    main()