
from pathlib import Path
from PIL import Image

DATASET = Path("data/multidisease")
extensions = {".jpg", ".jpeg", ".png"}

for split in ["train", "val", "test"]:
    print(f"\n--- {split.upper()} ---")

    for class_dir in sorted((DATASET / split).iterdir()):
        if not class_dir.is_dir():
            continue

        images = [
            f for f in class_dir.iterdir()
            if f.suffix.lower() in extensions
        ]

        corrupted = 0

        for image_path in images:
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception:
                corrupted += 1

        print(
            f"{class_dir.name}: "
            f"{len(images)} images | "
            f"Corrupted: {corrupted}"
        )