
import torch
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    balanced_accuracy_score,
    roc_auc_score
)

from model import LungCNN
from data_loader import val_loader, test_loader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LungCNN().to(DEVICE)

model.load_state_dict(
    torch.load(
        "fedavg_model.pth",
        map_location=DEVICE,
        weights_only=True
    )
)

model.eval()


def get_predictions(loader):
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)

            all_labels.extend(labels.numpy())
            all_probabilities.extend(
                probabilities[:, 1].cpu().numpy()
            )

    return np.array(all_labels), np.array(all_probabilities)


# Get validation predictions
val_labels, val_probs = get_predictions(val_loader)

# Select threshold using validation data
thresholds = np.arange(0.30, 0.71, 0.01)

best_threshold = 0.5
best_score = 0.0

for threshold in thresholds:
    predictions = (val_probs >= threshold).astype(int)

    score = balanced_accuracy_score(
        val_labels,
        predictions
    )

    if score > best_score:
        best_score = score
        best_threshold = threshold

print("\n===== Validation Threshold Selection =====")
print(f"Best Threshold: {best_threshold:.2f}")
print(f"Validation Balanced Accuracy: {best_score:.4f}")


# Evaluate on untouched test data
test_labels, test_probs = get_predictions(test_loader)

test_predictions = (
    test_probs >= best_threshold
).astype(int)

accuracy = accuracy_score(test_labels, test_predictions)
precision = precision_score(
    test_labels, test_predictions, zero_division=0
)
recall = recall_score(
    test_labels, test_predictions, zero_division=0
)
f1 = f1_score(
    test_labels, test_predictions, zero_division=0
)
balanced_accuracy = balanced_accuracy_score(
    test_labels, test_predictions
)
roc_auc = roc_auc_score(test_labels, test_probs)

cm = confusion_matrix(test_labels, test_predictions)

tn, fp, fn, tp = cm.ravel()

specificity = tn / (tn + fp)

print("\n===== Final FedAvg Test Results =====")
print(f"Threshold: {best_threshold:.2f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall/Sensitivity: {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)