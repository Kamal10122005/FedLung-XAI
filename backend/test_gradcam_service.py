
import sys
from pathlib import Path

from PIL import Image

# Add backend folder to Python path
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent)
)

from gradcam_service import generate_gradcam


PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_PATH = (
    PROJECT_ROOT
    / "data/multidisease/test/PNEUMONIA"
    / "person100_bacteria_475.jpeg"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results/backend_gradcam_test.png"
)


print("Loading X-ray...")

image = Image.open(IMAGE_PATH)

overlay, disease, confidence = generate_gradcam(image)

overlay.save(OUTPUT_PATH)

print("Grad-CAM generated successfully!")
print("Predicted Disease:", disease)
print("Confidence:", f"{confidence:.2f}%")
print("Saved:", OUTPUT_PATH)