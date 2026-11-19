
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
TEST_DIR = BASE_DIR / "data" / "real_fake_dataset" / "test"

incorrect_images = [
    ("fake", "fake_00096.jpg"),
    ("fake", "fake_00435.jpg"),
    ("fake", "fake_01664.jpg"),
    ("fake", "fake_02375.jpg"),
    ("fake", "fake_03184.jpg"),
    ("real", "real_00075.jpg"),
]

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

for ax, (label, filename) in zip(axes.flat, incorrect_images):
    image_path = TEST_DIR / label / filename

    if not image_path.exists():
        ax.set_title(f"Missing: {filename}")
        ax.axis("off")
        continue

    image = Image.open(image_path).convert("RGB")

    ax.imshow(image)
    ax.set_title(f"Actual: {label}\n{filename}")
    ax.axis("off")

plt.tight_layout()
plt.savefig(BASE_DIR / "reports" / "misclassified_images.png")
plt.show()

print("Misclassified images visualization saved.")
