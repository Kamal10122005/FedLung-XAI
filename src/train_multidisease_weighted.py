
import torch
import torch.nn as nn
import torch.optim as optim

from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from multidisease_loader import train_dataset, train_loader, val_loader
from multidisease_model import MultiDiseaseMobileNet
from sklearn.metrics import classification_report

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 8
LEARNING_RATE = 0.0001

print("Device:", DEVICE)

# Class weights from original training distribution
labels = [label for _, label in train_dataset.samples]
import numpy as np

classes = np.array(sorted(set(labels)))

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=labels
)

class_weights = torch.tensor(
    weights,
    dtype=torch.float32
).to(DEVICE)

print("Class weights:", class_weights)

# Model
model = MultiDiseaseMobileNet(
    num_classes=4,
    pretrained=True
).to(DEVICE)

# Unfreeze final feature layers
for parameter in model.model.features[-3:].parameters():
    parameter.requires_grad = True

criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

best_val_accuracy = 0
patience = 3
no_improvement = 0

for epoch in range(EPOCHS):

    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    train_accuracy = correct / total

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

    val_accuracy = val_correct / val_total

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Accuracy: {train_accuracy:.4f} | "
        f"Val Accuracy: {val_accuracy:.4f}"
    )

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        no_improvement = 0

        torch.save(
            model.state_dict(),
            "multidisease_weighted_best.pth"
        )

        print("Best model saved!")

    else:
        no_improvement += 1

    if no_improvement >= patience:
        print("Early stopping!")
        break

print("\nTraining completed!")
print("Best Validation Accuracy:", best_val_accuracy)