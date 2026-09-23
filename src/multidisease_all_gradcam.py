
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path

from multidisease_loader import test_dataset
from multidisease_model import MultiDiseaseMobileNet


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "multidisease_mobilenet_finetuned_best.pth"


# Load model
model = MultiDiseaseMobileNet(
    num_classes=4,
    pretrained=False
).to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )
)

model.eval()

classes = test_dataset.classes

print("Model loaded successfully!")
print("Classes:", classes)


# Grad-CAM activation storage
activations = None


def forward_hook(module, inputs, output):
    global activations

    activations = output

    # Keep gradients even if the tensor is not a leaf tensor
    activations.retain_grad()


# Last convolutional feature layer
target_layer = model.model.features[-1]

target_layer.register_forward_hook(forward_hook)


# Load prediction results
df = pd.read_csv(
    "results/multidisease_predictions_confidence.csv"
)


# Select one correct and one incorrect case per class
selected_cases = []

for disease in classes:

    correct = df[
        (df["actual"] == disease) &
        (df["correct"] == True)
    ]

    incorrect = df[
        (df["actual"] == disease) &
        (df["correct"] == False)
    ]

    if len(correct) > 0:
        selected_cases.append(
            ("correct", correct.iloc[0])
        )

    if len(incorrect) > 0:
        selected_cases.append(
            ("incorrect", incorrect.iloc[0])
        )


output_dir = Path("results/all_gradcam")
output_dir.mkdir(
    parents=True,
    exist_ok=True
)


# Generate Grad-CAM images
for case_type, row in selected_cases:

    print(
        f"\nProcessing: {row['actual']} - {case_type}"
    )

    image = Image.open(
        row["image_path"]
    ).convert("RGB")

    input_tensor = test_dataset.transform(
        image
    ).unsqueeze(0).to(DEVICE)

    # Enable gradients for input
    input_tensor.requires_grad_(True)

    model.zero_grad()

    # Forward pass
    output = model(input_tensor)

    predicted_class = output.argmax(
        dim=1
    ).item()

    score = output[0, predicted_class]

    # Backward pass
    score.backward()

    # Check gradients
    if activations is None or activations.grad is None:

        print("Gradients unavailable. Skipping image.")
        continue

    # Extract activations and gradients
    feature_maps = activations.detach()

    gradients = activations.grad.detach()

    # Global average pooling of gradients
    weights = gradients.mean(
        dim=(2, 3),
        keepdim=True
    )

    # Create class activation map
    cam = (
        weights * feature_maps
    ).sum(dim=1).squeeze()

    cam = torch.relu(cam)

    cam -= cam.min()

    cam /= cam.max() + 1e-8

    cam = cam.cpu().numpy()

    # Resize heatmap
    cam_image = Image.fromarray(
        np.uint8(cam * 255)
    ).resize(image.size)

    cam_array = np.array(
        cam_image
    ) / 255.0

    image_array = np.array(
        image
    ) / 255.0

    # Plot original image + heatmap
    plt.figure(figsize=(7, 6))

    plt.imshow(image_array)

    plt.imshow(
        cam_array,
        cmap="jet",
        alpha=0.45
    )

    plt.title(
        f"Actual: {row['actual']}\n"
        f"Predicted: {classes[predicted_class]}\n"
        f"Confidence: {row['confidence']:.2%}\n"
        f"Case: {case_type}"
    )

    plt.axis("off")

    plt.tight_layout()

    filename = (
        f"{row['actual']}_"
        f"{case_type}_"
        f"{classes[predicted_class]}.png"
    )

    save_path = output_dir / filename

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {save_path}")


print("\nAll-disease Grad-CAM completed!")