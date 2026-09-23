
import torch
import pandas as pd
from multidisease_loader import test_dataset, test_loader
from multidisease_model import MultiDiseaseMobileNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MultiDiseaseMobileNet(
    num_classes=4,
    pretrained=False
).to(DEVICE)

model.load_state_dict(
    torch.load(
        "multidisease_mobilenet_finetuned_best.pth",
        map_location=DEVICE,
        weights_only=True
    )
)

model.eval()

classes = test_dataset.classes
records = []
sample_index = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)

        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)

        confidences, predictions = torch.max(probabilities, dim=1)

        for i in range(len(labels)):
            actual_id = labels[i].item()
            predicted_id = predictions[i].item()

            records.append({
                "image_path": test_dataset.samples[sample_index][0],
                "actual": classes[actual_id],
                "predicted": classes[predicted_id],
                "confidence": round(confidences[i].item(), 4),
                "correct": actual_id == predicted_id
            })

            sample_index += 1

df = pd.DataFrame(records)

false_positives = df[
    (df["actual"] == "NORMAL") &
    (df["predicted"] == "PNEUMONIA")
]

print("\nNORMAL → PNEUMONIA confidence statistics:")
print(false_positives["confidence"].describe())

print("\nHighest-confidence wrong predictions:")
print(
    false_positives
    .sort_values("confidence", ascending=False)
    .head(15)
    .to_string(index=False)
)

df.to_csv(
    "results/multidisease_predictions_confidence.csv",
    index=False
)

print("\nConfidence analysis completed!")