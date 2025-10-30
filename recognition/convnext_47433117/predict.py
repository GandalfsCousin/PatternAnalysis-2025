import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from modules import custom_model, custom_small
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

    model = custom_small().to(Config.device)
    checkpoint = torch.load(Config.modelPath, map_location=Config.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(Config.device), labels.to(Config.device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)

    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)


if __name__ == "__main__":
    main()
