import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from modules import custom_small
from dataset import ADNI_Loader, Transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ConvNeXt model on ADNI test set")
    parser.add_argument('--data_root', type=str,
                        default="recognition/convnext_47433117/ADNI/AD_NC",
                        help="Path to ADNI dataset root")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for DataLoader")
    parser.add_argument('--model_path', type=str, required=True,
                        help="Path to saved model checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    class_names = ["AD", "NC"]

    test_dataset = ADNI_Loader(args.data_root, split="test", transform=Transforms.test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             pin_memory=True, num_workers=4)


    model = custom_small().to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)


    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])


    tp = cm[0, 0]
    fn = cm[0, 1]
    fp = cm[1, 0]
    tn = cm[1, 1]
    ad_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    ad_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    ad_f1 = 2 * ad_precision * ad_recall / (ad_precision + ad_recall) if (ad_precision + ad_recall) > 0 else 0.0

    # Print results
    print("\n=== Evaluation Results ===")
    print(f"Classes: {class_names}  (AD=0, NC=1)")
    print(f"Test Accuracy: {acc * 100:.2f}%")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall:    {recall:.4f}")
    print(f"Weighted F1-score:  {f1:.4f}\n")

    print("Confusion Matrix (rows = True, cols = Pred):")
    print(f"           Pred_AD    Pred_NC")
    print(f"True_AD   {cm[0, 0]:7d}   {cm[0, 1]:7d}")
    print(f"True_NC   {cm[1, 0]:7d}   {cm[1, 1]:7d}\n")

    print(f"AD (class 0) precision/recall/F1: {ad_precision:.4f} / {ad_recall:.4f} / {ad_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))


if __name__ == "__main__":
    main()
