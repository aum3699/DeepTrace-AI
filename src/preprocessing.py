
from pathlib import Path
from PIL import Image
import torch
from torchvision.models import EfficientNet_B0_Weights


# Official preprocessing transforms for EfficientNet-B0
weights = EfficientNet_B0_Weights.DEFAULT
preprocess = weights.transforms()


def preprocess_image(image_path):
    """Load an image and convert it into a model-ready tensor."""

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")

    tensor = preprocess(image)

    return tensor


def preprocess_folder(folder_path):
    """Preprocess all valid images inside a folder."""

    folder_path = Path(folder_path)

    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    tensors = []
    processed_files = []

    for image_path in folder_path.rglob("*"):
        if image_path.suffix.lower() not in image_extensions:
            continue

        try:
            tensor = preprocess_image(image_path)
            tensors.append(tensor)
            processed_files.append(str(image_path))

        except Exception as error:
            print(f"Skipped {image_path.name}: {error}")

    return tensors, processed_files
