
import torch
import pandas as pd
from pathlib import Path
from multidisease_loader import test_dataset, test_loader
from multidisease_model import MultiDiseaseMobileNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MultiDiseaseMobileNet(num_classes=4, pretrained=False).to(DEVICE)
model.load_state_dict(
    torch.load(
        "multidisease_mobilenet_finetuned_best.pth",
        map_location=DEVICE,
        weights_only=True
    )
)
model.eval()

class_names = test_dataset.classes
records = []

with torch.no_grad():
    sample_index = 0

    for images, labels in test_loader:
        images = images.to(DEVICE)

        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)

        for i in range(len(labels)):
            actual = labels[i].item()
            predicted = predictions[i].item()

            image_path = test_dataset.samples[sample_index][0]

            records.append({
                "image_path": image_path,
                "actual": class_names[actual],
                "predicted": class_names[predicted],
                "correct": actual == predicted
            })

            sample_index += 1

df = pd.DataFrame(records)

Path("results").mkdir(exist_ok=True)
df.to_csv("results/multidisease_test_predictions.csv", index=False)

print("\nTotal test images:", len(df))
print("Correct predictions:", df["correct"].sum())
print("Wrong predictions:", (~df["correct"]).sum())

print("\nMost common error pairs:")
errors = df[df["correct"] == False]

print(
    errors.groupby(["actual", "predicted"])
    .size()
    .sort_values(ascending=False)
)

print("\nNORMAL predicted as PNEUMONIA:")
normal_errors = df[
    (df["actual"] == "NORMAL") &
    (df["predicted"] == "PNEUMONIA")
]

print("Count:", len(normal_errors))
print(normal_errors.head(10).to_string(index=False))

print("\nError analysis completed!")
