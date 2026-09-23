
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from model import LungCNN


# Check image path
if len(sys.argv) < 2:
    print("Usage: python src/inference_xai.py <image_path>")
    sys.exit(1)

image_path = Path(sys.argv[1])

if not image_path.exists():
    raise FileNotFoundError(f"Image not found: {image_path}")


# Device
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Class names
class_names = ["NORMAL", "PNEUMONIA"]


# Image transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Load image
original_image = Image.open(image_path).convert("RGB")

image = transform(original_image)

input_image = image.unsqueeze(0).to(DEVICE)


# Load model
model = LungCNN().to(DEVICE)

model.load_state_dict(
    torch.load("fedavg_model.pth", map_location=DEVICE)
)

model.eval()


# Grad-CAM variables
activations = None
gradients = None


# Target convolutional layer
target_layer = model.features[6]


def forward_hook(module, inputs, output):
    global activations
    activations = output


def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]


forward_handle = target_layer.register_forward_hook(
    forward_hook
)

backward_handle = target_layer.register_full_backward_hook(
    backward_hook
)


# Forward pass
output = model(input_image)

probabilities = torch.softmax(output, dim=1)

predicted_class = output.argmax(dim=1).item()

confidence = probabilities[0, predicted_class].item() * 100


print("\nPrediction Results")
print("------------------")
print("Predicted Class:", class_names[predicted_class])
print(f"Confidence: {confidence:.2f}%")


# Backward pass
model.zero_grad()

output[0, predicted_class].backward()


# Generate Grad-CAM
weights = gradients.mean(
    dim=(2, 3),
    keepdim=True
)

cam = (weights * activations).sum(dim=1).squeeze()

cam = F.relu(cam)

cam = cam / (cam.max() + 1e-8)

cam = cam.detach().cpu().numpy()


# Resize heatmap
cam_tensor = torch.tensor(cam).unsqueeze(0).unsqueeze(0)

cam_resized = F.interpolate(
    cam_tensor,
    size=(128, 128),
    mode="bilinear",
    align_corners=False
).squeeze().numpy()


# Prepare original image
display_image = original_image.resize((128, 128))

original = torch.tensor(
    list(display_image.getdata()),
    dtype=torch.float32
).reshape(128, 128, 3).numpy() / 255.0


# Create heatmap
heatmap = plt.cm.jet(cam_resized)[:, :, :3]


# Create overlay
overlay = (0.6 * original) + (0.4 * heatmap)

overlay = overlay.clip(0, 1)


# Create figure
fig, axes = plt.subplots(
    1, 3,
    figsize=(15, 5)
)


axes[0].imshow(original)

axes[0].set_title("Original X-ray")

axes[0].axis("off")


axes[1].imshow(heatmap)

axes[1].set_title("Grad-CAM Heatmap")

axes[1].axis("off")


axes[2].imshow(overlay)

axes[2].set_title(
    f"Prediction: {class_names[predicted_class]}\n"
    f"Confidence: {confidence:.2f}%"
)

axes[2].axis("off")


fig.suptitle(
    "FedLung-XAI Image Explanation",
    fontsize=14
)

plt.tight_layout()


# Save explanation
output_filename = "gradcam_uploaded_image.png"

plt.savefig(
    output_filename,
    dpi=300,
    bbox_inches="tight"
)

print("\nExplanation saved as:", output_filename)


# Remove hooks
forward_handle.remove()
backward_handle.remove()

plt.show()