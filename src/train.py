import torch
import torch.nn as nn
import torch.optim as optim

from model import LungCNN
from data_loader import train_loader, val_loader, train_dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LungCNN().to(device)

# Calculate class weights
class_counts = torch.bincount(
    torch.tensor(train_dataset.targets)
)

class_weights = len(train_dataset) / (
    len(class_counts) * class_counts.float()
)

class_weights = class_weights.to(device)

print("Class counts:", class_counts.tolist())
print("Class weights:", class_weights.tolist())

# Weighted loss
criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

EPOCHS = 10
best_val_loss = float("inf")

for epoch in range(EPOCHS):

    # Training
    model.train()
    train_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()
    
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)

    val_accuracy = 100 * correct / total

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.2f}%"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            "best_model.pth"
        )

        print("Best model saved!")


print("Training completed!")