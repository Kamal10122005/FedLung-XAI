
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import classification_report
from collections import Counter

from multidisease_loader import train_dataset, train_loader, val_loader
from multidisease_model import MultiDiseaseMobileNet


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 8
LEARNING_RATE = 0.0001

print("Device:", DEVICE)

model = MultiDiseaseMobileNet(
    num_classes=4,
    pretrained=True
).to(DEVICE)

# Fine-tune the final feature layers
for parameter in model.model.features[-3:].parameters():
    parameter.requires_grad = True

# Class-weighted sampling
class_counts = Counter(train_dataset.targets)
sample_weights = [
    1.0 / class_counts[label]
    for label in train_dataset.targets
]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

# Use a balanced sampler instead of shuffle
from torch.utils.data import DataLoader

balanced_loader = DataLoader(
    train_dataset,
    batch_size=32,
    sampler=sampler,
    num_workers=0
)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

best_val_accuracy = 0.0
patience = 3
no_improvement = 0

for epoch in range(EPOCHS):

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in balanced_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)

        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
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
            predictions = torch.argmax(outputs, dim=1)

            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

    val_accuracy = val_correct / val_total

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.4f} | "
        f"Val Accuracy: {val_accuracy:.4f}"
    )

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        no_improvement = 0

        torch.save(
            model.state_dict(),
            "multidisease_mobilenet_finetuned_best.pth"
        )

        print("Best model saved!")
    else:
        no_improvement += 1

        if no_improvement >= patience:
            print("Early stopping triggered!")
            break

print("\nTraining completed!")
print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")

# Load best model
model.load_state_dict(
    torch.load(
        "multidisease_mobilenet_finetuned_best.pth",
        map_location=DEVICE,
        weights_only=True
    )
)

model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)

        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.numpy())

print("\nValidation Classification Report:")
print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=train_dataset.classes,
        zero_division=0
    )
)