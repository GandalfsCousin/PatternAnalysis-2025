import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules import small_model
from dataset import ADNI_Loader, Transforms


class Config:
    dataRoot = "recognition/convnext_47433117/ADNI/AD_NC"
    batchSize = 16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    modelPath = "checkpoints/77.pth"


def main():
    test_dataset = ADNI_Loader(Config.dataRoot, split="test", transform=Transforms.test_transform)
    test_loader = DataLoader(
        test_dataset, batch_size=Config.batchSize, shuffle=False,
        pin_memory=True, num_workers=4
    )

    model = small_model().to(Config.device)
    checkpoint = torch.load(Config.modelPath, map_location=Config.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(Config.device), labels.to(Config.device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"\nTest Accuracy: {acc * 100:.2f}% ({correct}/{total})")


if __name__ == "__main__":
    main()
