import sys
import torch
from PIL import Image
from torchvision import transforms

sys.path.append("src")

from multidisease_model import MultiDiseaseMobileNet

MODEL_PATH = "multidisease_mobilenet_finetuned_best.pth"

CLASSES = ["COVID19", "NORMAL", "PNEUMONIA", "TUBERCULOSIS"]

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.456],
        std=[0.229, 0.224, 0.225]
    )
])

model = MultiDiseaseMobileNet(
    num_classes=4,
    pretrained=False
)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
)

model.eval()

image_path = (
    "data/multidisease/test/PNEUMONIA/"
    "person100_bacteria_475.jpeg"
)

image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)
    probabilities = torch.softmax(output, dim=1)
    predicted_index = torch.argmax(probabilities, dim=1).item()

print("Predicted Disease:", CLASSES[predicted_index])
print(
    "Confidence:",
    f"{probabilities[0][predicted_index].item() * 100:.2f}%"
)