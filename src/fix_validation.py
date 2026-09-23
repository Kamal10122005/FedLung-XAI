
from pathlib import Path
import shutil
import random

random.seed(42)

DATASET = Path("data/chest_xray")
TRAIN = DATASET / "train"
VAL = DATASET / "val"

VAL_RATIO = 0.15

for class_name in ["NORMAL", "PNEUMONIA"]:
    train_dir = TRAIN / class_name
    val_dir = VAL / class_name

    val_dir.mkdir(parents=True, exist_ok=True)

    # Move old validation images into training
    for image in val_dir.iterdir():
        if image.is_file():
            destination = train_dir / image.name

            if not destination.exists():
                shutil.move(str(image), str(destination))

    # Select images for new validation split
    images = [
        image for image in train_dir.iterdir()
        if image.is_file()
    ]

    random.shuffle(images)

    val_count = int(len(images) * VAL_RATIO)
    selected_images = images[:val_count]

    for image in selected_images:
        shutil.move(str(image), str(val_dir / image.name))

    print(f"{class_name}: {len(selected_images)} validation images")

print("\nValidation split completed successfully!")