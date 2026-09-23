import sys
import torch

sys.path.append("src")

from multidisease_model import MultiDiseaseMobileNet

MODEL_PATH = "multidisease_mobilenet_finetuned_best.pth"

model = MultiDiseaseMobileNet(num_classes=4, pretrained=False)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
)

model.eval()

print("Model loaded successfully!")
print("Classes: COVID19, NORMAL, PNEUMONIA, TUBERCULOSIS")