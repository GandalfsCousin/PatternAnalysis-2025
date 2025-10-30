import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import argparse

from modules import custom_model, custom_small
from dataset import ADNI_Loader, Transforms

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ConvNeXt model on ADNI test set")
    parser.add_argument('--data_root', type=str, default="recognition/convnext_47433117/ADNI/AD_NC", help="Path to ADNI dataset root")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for DataLoader")
    parser.add_argument('--model_path', type=str, required=True, help="Path to saved model checkpoint")
    return parser.parse_args()

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    test_dataset = ADNI_Loader(args.data_root, split="test", transform=Transforms.test_transform)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        pin_memory=True, num_workers=4
    )

    model = custom_small().to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
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
