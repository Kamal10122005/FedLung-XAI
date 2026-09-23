
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_PATH))

from multidisease_model import MultiDiseaseMobileNet


DEVICE = torch.device("cpu")

MODEL_PATH = (
    PROJECT_ROOT / "multidisease_mobilenet_finetuned_best.pth"
)

CLASSES = [
    "COVID19",
    "NORMAL",
    "PNEUMONIA",
    "TUBERCULOSIS"
]


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Load trained model
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

print("Grad-CAM model loaded successfully!")


def describe_attention_region(cam):
    """
    Estimate the image region where Grad-CAM
    activation is strongest.

    This describes model attention, not a confirmed
    anatomical lesion or disease location.
    """

    height, width = cam.shape

    # Select stronger activation areas
    threshold = np.percentile(cam, 80)

    mask = cam >= threshold

    coordinates = np.argwhere(mask)

    if len(coordinates) == 0:
        return {
            "region": "Not clearly localized",
            "activation_strength": 0.0,
            "attention_percentage": 0.0
        }

    y_coordinates = coordinates[:, 0]
    x_coordinates = coordinates[:, 1]

    center_y = float(np.mean(y_coordinates))
    center_x = float(np.mean(x_coordinates))

    # Horizontal location
    if center_x < width / 3:
        horizontal = "left"
    elif center_x > (2 * width) / 3:
        horizontal = "right"
    else:
        horizontal = "central"

    # Vertical location
    if center_y < height / 3:
        vertical = "upper"
    elif center_y > (2 * height) / 3:
        vertical = "lower"
    else:
        vertical = "middle"

    region = f"{vertical}-{horizontal}"

    activation_strength = float(
        np.mean(cam[mask])
    )

    attention_percentage = float(
        (np.sum(mask) / mask.size) * 100
    )

    return {
        "region": region,
        "activation_strength": round(
            activation_strength, 2
        ),
        "attention_percentage": round(
            attention_percentage, 2
        )
    }


def create_dynamic_explanation(
    disease,
    confidence,
    region_info
):
    """
    Create a concise, point-wise explanation
    using the current prediction and Grad-CAM statistics.
    """

    region = region_info["region"]
    strength = region_info["activation_strength"]
    percentage = region_info["attention_percentage"]

    explanation = (
        f"• Prediction: {disease}\n"
        f"• Confidence: {confidence:.2f}%\n"
        f"• Attention Region: {region}\n"
        f"• Activation Strength: {strength:.2f}\n"
        f"• Attention Coverage: {percentage:.2f}%\n"
        f"• Interpretation: Highlighted areas influenced "
        f"the model's prediction.\n"
        f"• Important Note: These areas are not confirmed "
        f"disease locations or a medical diagnosis."
    )

    return explanation


def generate_gradcam(image):

    original_image = image.convert("RGB")

    input_tensor = transform(
        original_image
    ).unsqueeze(0).to(DEVICE)

    input_tensor.requires_grad_(True)

    activations = {}

    target_layer = model.model.features[-1]

    def forward_hook(module, inputs, output):
        activations["value"] = output
        output.retain_grad()

    hook = target_layer.register_forward_hook(
        forward_hook
    )

    model.zero_grad()

    # Forward pass
    output = model(input_tensor)

    probabilities = torch.softmax(
        output,
        dim=1
    )

    predicted_index = torch.argmax(
        probabilities,
        dim=1
    ).item()

    score = output[0, predicted_index]

    # Backward pass
    score.backward()

    feature_maps = activations["value"].detach()

    gradients = activations["value"].grad.detach()

    # Calculate Grad-CAM weights
    weights = gradients.mean(
        dim=(2, 3),
        keepdim=True
    )

    cam = (
        weights * feature_maps
    ).sum(dim=1).squeeze()

    cam = torch.relu(cam)

    cam = cam.cpu().numpy()

    # Normalize heatmap
    cam = cam - cam.min()

    if cam.max() > 0:
        cam = cam / cam.max()

    # Extract dynamic region information
    region_info = describe_attention_region(cam)

    cam_uint8 = (cam * 255).astype(np.uint8)

    # Convert original image to array
    original_array = np.array(
        original_image
    )

    height, width = original_array.shape[:2]

    # Resize heatmap
    heatmap = cv2.resize(
        cam_uint8,
        (width, height)
    )

    heatmap_color = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )

    # Overlay heatmap
    overlay = cv2.addWeighted(
        original_array,
        0.6,
        heatmap_color,
        0.4,
        0
    )

    hook.remove()

    overlay_image = Image.fromarray(
        overlay
    )

    disease = CLASSES[predicted_index]

    confidence = (
        probabilities[0][predicted_index].item()
        * 100
    )

    explanation = create_dynamic_explanation(
        disease,
        confidence,
        region_info
    )

    return (
        overlay_image,
        disease,
        confidence,
        region_info,
        explanation
    )
