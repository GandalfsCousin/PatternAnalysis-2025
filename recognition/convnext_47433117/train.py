import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm
import matplotlib.pyplot as plt

from modules import small_model, custom_model
from dataset import ADNI_Loader, Transforms

 

class Config:
    dataRoot = "recognition/convnext_47433117/ADNI/AD_NC"
    batchSize = 32
    epochs = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    learningRate = 2e-4
    weightDecay = 0.05
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


def plot_metrics(train_losses, val_losses, train_accs, val_accs, save_dir):
    """Plot training and validation metrics"""
    epochs = range(1, len(train_losses) + 1)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot losses
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracies
    ax2.plot(epochs, train_accs, 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training vs Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(save_dir, 'training_metrics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining metrics plot saved to: {plot_path}")
    
    plt.show()


def main():
    train_dataset = ADNI_Loader(Config.dataRoot, split="train", transform=Transforms.aggressive_train_transform)
    test_dataset = ADNI_Loader(Config.dataRoot, split="test", transform=Transforms.test_transform)
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=Config.batchSize, shuffle=True, pin_memory=True, num_workers=6)
    test_loader = DataLoader(test_dataset, batch_size=Config.batchSize, shuffle=False, pin_memory=True, num_workers=6)
    

    model = custom_model().to(Config.device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.learningRate, weight_decay=Config.weightDecay)

    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=5)
    cosine = CosineAnnealingLR(optimizer, T_max=Config.epochs - 5)
    scheduler = SequentialLR(optimizer, 
                            schedulers=[warmup, cosine], 
                            milestones=[5])

    best_acc = 0.0
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(Config.epochs):
        print(f"===== Epoch {epoch+1}/{Config.epochs} =====")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, Config.device)
        val_loss, val_acc = validate(model, test_loader, criterion, Config.device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  LR: {current_lr:.6f}\n")

        if val_acc > best_acc:
            best_acc = val_acc
            print(f"  Saving best model with accuracy: {best_acc:.2f}%")

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
    
    print(f"\n{'='*50}")
    print(f"Training Complete!")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"{'='*50}\n")
    
    # Plot the metrics
    plot_metrics(train_losses, val_losses, train_accs, val_accs, Config.saveDir)

if __name__ == "__main__":
    main()