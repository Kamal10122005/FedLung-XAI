
from pathlib import Path

DATASET = Path("data/multidisease")

for split in ["train", "val", "test"]:
    print(f"\n{split.upper()} DISTRIBUTION")

    total = 0

    for class_folder in sorted((DATASET / split).iterdir()):
        if class_folder.is_dir():
            count = len(list(class_folder.glob("*")))
            total += count
            print(f"{class_folder.name}: {count}")

    print("Total:", total)