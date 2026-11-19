from pathlib import Path
from PIL import Image
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "real_fake_dataset"

print("\n===== DEEPTRACE DATASET INSPECTION =====")

for split in ["train", "val", "test"]:
    print(f"\n{split.upper()} DATASET")

    for label in ["fake", "real"]:
        folder = DATA_DIR / split / label

        images = list(folder.glob("*"))

        valid = 0
        invalid = 0

        for image_path in images:
            try:
                with Image.open(image_path) as img:
                    img.verify()
                valid += 1
            except Exception:
                invalid += 1

        print(f"{label}:")
        print("Total images:", len(images))
        print("Valid images:", valid)
        print("Invalid images:", invalid)

print("\n===== INSPECTION COMPLETED =====")