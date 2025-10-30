import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt
from modules import custom_model, custom_small
from dataset import ADNI_Loader, Transforms
import numpy as np

os.makedirs("checkpoints", exist_ok=True)

class Config:
    dataRoot = "recognition/convnext_47433117/ADNI/AD_NC"
    batchSize = 32
    epochs = 50
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    learningRate = 4e-4
    weightDecay = 0.05
    dropPathRate = 0.25
    classifierDropout = 0.3
    labelSmoothing = 0.02
    saveDir = "checkpoints"


def train_val_split(dataset, val_ratio=0.1):
    unique_patients = np.unique(dataset.patient_ids)
    np.random.shuffle(unique_patients)

    n_val = int(len(unique_patients) * val_ratio)
    val_patients = unique_patients[:n_val]
    train_patients = unique_patients[n_val:]

    train_idx = [i for i, pid in enumerate(dataset.patient_ids) if pid in train_patients]
    val_idx = [i for i, pid in enumerate(dataset.patient_ids) if pid in val_patients]

    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total, correct, running_loss = 0, 0, 0
    loop = tqdm(loader, desc="Training", leave=True)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * images.size(0)
        _, pred = outputs.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()
        loop.set_postfix(loss=loss.item(), acc=100.*correct/total)
    return running_loss / total, 100.*correct/total

def validate(model, loader, criterion, device):
    model.eval()
    total, correct, running_loss = 0, 0, 0
    with torch.no_grad(), torch.amp.autocast('cuda'):
        loop = tqdm(loader, desc="Validation", leave=True)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
            loop.set_postfix(loss=loss.item(), acc=100.*correct/total)
    return running_loss / total, 100.*correct/total



def plot_metrics(train_losses, val_losses, lrs, save_dir):
    epochs = range(1, len(train_losses)+1)

    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(epochs, lrs, 'g-', label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.title('Learning Rate Schedule')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'lr_curve.png'), dpi=300)
    plt.close()



def main():
    dataset = ADNI_Loader(Config.dataRoot, split="train", transform=Transforms.train_transform)
    train_subset, val_subset = train_val_split(dataset)

    train_loader = DataLoader(train_subset, batch_size=Config.batchSize, shuffle=True, num_workers=6)
    val_loader = DataLoader(val_subset, batch_size=Config.batchSize, shuffle=False, num_workers=6)

    model = custom_small(drop_path_rate=Config.dropPathRate, classifier_dropout=Config.classifierDropout).to(Config.device)
    criterion = nn.CrossEntropyLoss(label_smoothing=Config.labelSmoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.learningRate, weight_decay=Config.weightDecay)
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.epochs)
    scaler = torch.amp.GradScaler('cuda')

    best_acc = 0.0
    train_losses, val_losses, lrs = [], [], []

    for epoch in range(Config.epochs):
        print(f"\n===== Epoch {epoch+1}/{Config.epochs} | LR: {optimizer.param_groups[0]['lr']:.6f} =====")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, Config.device)
        val_loss, val_acc = validate(model, val_loader, criterion, Config.device)
        scheduler.step()
        lrs.append(optimizer.param_groups[0]['lr'])

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f" Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f" Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, os.path.join(Config.saveDir, 'best_model.pth'))

    plot_metrics(train_losses, val_losses, lrs, Config.saveDir)
    print(f"\nTraining complete! Best Val Accuracy: {best_acc:.2f}%")

if __name__=="__main__":
    main()
