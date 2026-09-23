
from pathlib import Path
import random
import shutil

random.seed(42)

DATASET = Path("data/multidisease")
CLASSES = ["COVID19", "NORMAL", "PNEUMONIA", "TUBERCULOSIS"]
VAL_RATIO = 0.15

for class_name in CLASSES:
    train_dir = DATASET / "train" / class_name
    val_dir = DATASET / "val" / class_name

    val_dir.mkdir(parents=True, exist_ok=True)

    # Move existing validation images back to train
    for image in val_dir.iterdir():
        if image.is_file():
            shutil.move(str(image), str(train_dir / image.name))

    images = [
        image for image in train_dir.iterdir()
        if image.is_file()
    ]

    random.shuffle(images)
    val_count = int(len(images) * VAL_RATIO)

    for image in images[:val_count]:
        shutil.move(str(image), str(val_dir / image.name))

    print(
        f"{class_name}: "
        f"Train={len(list(train_dir.iterdir()))}, "
        f"Val={len(list(val_dir.iterdir()))}"
    )

print("Validation split completed!")