
from pathlib import Path
from PIL import Image

DATASET = Path("data/chest_xray")
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

total_images = 0
corrupted_images = []

for split in ["train", "val", "test"]:
    for class_name in ["NORMAL", "PNEUMONIA"]:
        folder = DATASET / split / class_name

        for image_path in folder.iterdir():
            if image_path.suffix.lower() not in VALID_EXTENSIONS:
                continue

            total_images += 1

            try:
                with Image.open(image_path) as image:
                    image.verify()

            except Exception:
                corrupted_images.append(str(image_path))

print(f"Total images checked: {total_images}")
print(f"Corrupted images: {len(corrupted_images)}")

if corrupted_images:
    print("\nCorrupted files:")
    for image in corrupted_images:
        print(image)
else:
    print("\nAll images are valid!")