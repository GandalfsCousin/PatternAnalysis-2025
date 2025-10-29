import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules import small_model, custom_model
from dataset import ADNI_Loader, Transforms

 

class Config:
    dataRoot = "recognition/convnext_47433117/ADNI/AD_NC"
    batchSize = 16
    epochs = 50
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    learningRate = 5e-4
    weightDecay = 0.05
    warmupEpochs = 5
    saveDir = "checkpoints"


def train_epoch(model, loader, criterion, optimizer, device):
    """Run the training for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(loader, desc="Training", leave=False)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        loop.set_postfix(loss=loss.item(), acc=100.*correct/total)
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        loop = tqdm(loader, desc="Validation!!", leave=False)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            loop.set_postfix(loss=loss.item(), acc=100.*correct/total)
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def main():
    train_dataset = ADNI_Loader(Config.dataRoot, split="train", transform=Transforms.train_transform)
    test_dataset = ADNI_Loader(Config.dataRoot, split="test", transform=Transforms.test_transform)
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=Config.batchSize, shuffle=True, pin_memory=True, num_workers=6)
    test_loader = DataLoader(test_dataset, batch_size=Config.batchSize, shuffle=False, pin_memory=True, num_workers=6)
    

    model = small_model().to(Config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.learningRate, weight_decay=Config.weightDecay)

    best_acc = 0.0

    for epoch in range(Config.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, Config.device)
        val_loss, val_acc = validate(model, test_loader, criterion, Config.device)

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, os.path.join(Config.saveDir, 'best_model.pth'))

        if (epoch + 1) % 5 == 0: # save every 5 cause
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(Config.saveDir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    print(f"Best validation accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()