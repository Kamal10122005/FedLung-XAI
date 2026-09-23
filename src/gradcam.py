
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from model import LungCNN
from data_loader import test_dataset


# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load trained model
model = LungCNN().to(DEVICE)

model.load_state_dict(
    torch.load("fedavg_model.pth", map_location=DEVICE)
)

model.eval()

class_names = test_dataset.classes


# Find a misclassified image
selected_index = None

for i in range(len(test_dataset)):
    candidate_image, candidate_label = test_dataset[i]

    candidate_input = candidate_image.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        candidate_output = model(candidate_input)

    candidate_prediction = candidate_output.argmax(dim=1).item()

    # Select misclassified image
    if candidate_prediction != candidate_label:
        selected_index = i
        break


if selected_index is None:
    raise RuntimeError("No misclassified image found.")


# Select image
image, label = test_dataset[selected_index]

input_image = image.unsqueeze(0).to(DEVICE)

print("Selected Image Index:", selected_index)


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


target_layer.register_forward_hook(forward_hook)

target_layer.register_full_backward_hook(backward_hook)


# Forward pass
output = model(input_image)

probabilities = torch.softmax(output, dim=1)

predicted_class = output.argmax(dim=1).item()

confidence = probabilities[0, predicted_class].item() * 100

actual_class = label


# Print prediction details
print("Actual Class:", class_names[actual_class])

print("Predicted Class:", class_names[predicted_class])

print(f"Confidence: {confidence:.2f}%")


# Backward pass
model.zero_grad()

output[0, predicted_class].backward()


# Generate Grad-CAM
weights = gradients.mean(dim=(2, 3), keepdim=True)

cam = (weights * activations).sum(dim=1).squeeze()

cam = F.relu(cam)

cam = cam / (cam.max() + 1e-8)

cam = cam.detach().cpu().numpy()


# Resize CAM
cam_tensor = torch.tensor(cam).unsqueeze(0).unsqueeze(0)

cam_resized = F.interpolate(
    cam_tensor,
    size=(128, 128),
    mode="bilinear",
    align_corners=False
).squeeze().numpy()


# Denormalize original image
original = image.permute(1, 2, 0).numpy()

mean = [0.485, 0.456, 0.406]

std = [0.229, 0.224, 0.225]

original = original * std + mean

original = original.clip(0, 1)


# Create heatmap
heatmap = plt.cm.jet(cam_resized)[:, :, :3]


# Create overlay
overlay = (0.6 * original) + (0.4 * heatmap)

overlay = overlay.clip(0, 1)


# Create figure
fig, axes = plt.subplots(1, 3, figsize=(15, 5))


# Original X-ray
axes[0].imshow(original)

axes[0].set_title(
    f"Original X-ray\nActual: {class_names[actual_class]}"
)

axes[0].axis("off")


# Heatmap
axes[1].imshow(heatmap)

axes[1].set_title("Grad-CAM Heatmap")

axes[1].axis("off")


# Overlay
axes[2].imshow(overlay)

axes[2].set_title(
    f"Predicted: {class_names[predicted_class]}\n"
    f"Confidence: {confidence:.2f}%"
)

axes[2].axis("off")


# Main title
fig.suptitle(
    f"Grad-CAM Misclassification | "
    f"Actual: {class_names[actual_class]} | "
    f"Predicted: {class_names[predicted_class]}",
    fontsize=14
)


plt.tight_layout()


# Save image
output_filename = "gradcam_misclassified.png"

plt.savefig(
    output_filename,
    dpi=300,
    bbox_inches="tight"
)

print(f"Grad-CAM image saved as {output_filename}")


# Display figure
plt.show()