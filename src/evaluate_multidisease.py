
import torch
from sklearn.metrics import classification_report, confusion_matrix

from multidisease_loader import test_loader, train_dataset
from multidisease_model import MultiDiseaseMobileNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MultiDiseaseMobileNet(
    num_classes=4,
    pretrained=False
).to(DEVICE)

model.load_state_dict(
    torch.load(
        "multidisease_weighted_best.pth",
        map_location=DEVICE,
        weights_only=True
    )
)

model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)

        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.numpy())

print("Test Classification Report:")
print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=train_dataset.classes,
        zero_division=0
    )
)

print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_predictions))
