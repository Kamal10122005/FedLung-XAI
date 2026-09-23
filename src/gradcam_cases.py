
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from model import LungCNN
from data_loader import test_dataset


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = LungCNN().to(DEVICE)

model.load_state_dict(
    torch.load("fedavg_model.pth", map_location=DEVICE)
)

model.eval()

class_names = test_dataset.classes


# Find required cases
cases = {
    "normal_correct": (0, 0),
    "pneumonia_correct": (1, 1),
    "false_positive": (0, 1),
    "false_negative": (1, 0)
}

selected_cases = {}

for i in range(len(test_dataset)):

    image, actual_label = test_dataset[i]

    input_image = image.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_image)

    predicted_label = output.argmax(dim=1).item()

    for case_name, (required_actual, required_prediction) in cases.items():

        if case_name in selected_cases:
            continue

        if (
            actual_label == required_actual
            and predicted_label == required_prediction
        ):
            selected_cases[case_name] = i


print("\nSelected Cases:")

for case_name, index in selected_cases.items():
    print(case_name, "-> Image Index:", index)


# Generate Grad-CAM
def generate_gradcam(image, actual_label, case_name):

    input_image = image.unsqueeze(0).to(DEVICE)

    activations = None
    gradients = None

    def forward_hook(module, inputs, output):
        nonlocal activations
        activations = output

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    target_layer = model.features[6]

    forward_handle = target_layer.register_forward_hook(
        forward_hook
    )

    backward_handle = target_layer.register_full_backward_hook(
        backward_hook
    )

    output = model(input_image)

    probabilities = torch.softmax(output, dim=1)

    predicted_class = output.argmax(dim=1).item()

    confidence = (
        probabilities[0, predicted_class].item() * 100
    )

    model.zero_grad()

    output[0, predicted_class].backward()

    weights = gradients.mean(
        dim=(2, 3),
        keepdim=True
    )

    cam = (weights * activations).sum(dim=1).squeeze()

    cam = F.relu(cam)

    cam = cam / (cam.max() + 1e-8)

    cam = cam.detach().cpu().numpy()

    cam_tensor = torch.tensor(cam).unsqueeze(0).unsqueeze(0)

    cam_resized = F.interpolate(
        cam_tensor,
        size=(128, 128),
        mode="bilinear",
        align_corners=False
    ).squeeze().numpy()

    # Denormalize image
    original = image.permute(1, 2, 0).numpy()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    original = original * std + mean

    original = original.clip(0, 1)

    heatmap = plt.cm.jet(cam_resized)[:, :, :3]

    overlay = (0.6 * original) + (0.4 * heatmap)

    overlay = overlay.clip(0, 1)

    # Create figure
    fig, axes = plt.subplots(
        1, 3,
        figsize=(15, 5)
    )

    axes[0].imshow(original)

    axes[0].set_title(
        f"Original X-ray\n"
        f"Actual: {class_names[actual_label]}"
    )

    axes[0].axis("off")

    axes[1].imshow(heatmap)

    axes[1].set_title("Grad-CAM Heatmap")

    axes[1].axis("off")

    axes[2].imshow(overlay)

    axes[2].set_title(
        f"Predicted: {class_names[predicted_class]}\n"
        f"Confidence: {confidence:.2f}%"
    )

    axes[2].axis("off")

    fig.suptitle(
        f"{case_name}\n"
        f"Actual: {class_names[actual_label]} | "
        f"Predicted: {class_names[predicted_class]}",
        fontsize=14
    )

    plt.tight_layout()

    output_filename = f"gradcam_{case_name}.png"

    plt.savefig(
        output_filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"Saved: {output_filename} | "
        f"Confidence: {confidence:.2f}%"
    )

    forward_handle.remove()
    backward_handle.remove()


# Generate all available cases
for case_name, index in selected_cases.items():

    image, actual_label = test_dataset[index]

    generate_gradcam(
        image,
        actual_label,
        case_name
    )


print("\nXAI generation completed!")