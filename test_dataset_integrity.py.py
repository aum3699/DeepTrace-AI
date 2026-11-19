
from pathlib import Path
from PIL import Image

DATASET_DIR = Path("data/real_fake_dataset")

splits = ["train", "val", "test"]
classes = ["real", "fake"]

total_valid = 0
total_invalid = 0

print("\n--- DEEPTRACE DATASET INTEGRITY CHECK ---")

for split in splits:
    for label in classes:
        folder = DATASET_DIR / split / label

        valid = 0
        invalid = 0

        for file in folder.iterdir():
            if file.is_file():
                try:
                    with Image.open(file) as img:
                        img.verify()
                    valid += 1
                except Exception as e:
                    invalid += 1
                    print(f"Invalid: {file} | {e}")

        total_valid += valid
        total_invalid += invalid

        print(f"{split}/{label}:")
        print(f"  Valid images: {valid}")
        print(f"  Invalid images: {invalid}")

print("\n--- FINAL REPORT ---")
print("Total valid images:", total_valid)
print("Total invalid images:", total_invalid)

if total_valid == 200 and total_invalid == 0:
    print("SUCCESS: All 200 images passed integrity checks!")
else:
    print("WARNING: Dataset needs further inspection.")
