
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "real_fake_dataset" / "test"
MODEL_PATH = BASE_DIR / "models" / "deeptrace_efficientnet_b0.pth"

DEVICE = torch.device("cpu")
IMAGE_SIZE = 224

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
classes = checkpoint["classes"]

model = models.efficientnet_b0(weights=None)
input_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(input_features, len(classes))

model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

print("\n===== DEEPTRACE ERROR ANALYSIS =====")
print("Classes:", classes)

correct = 0
incorrect = 0

with torch.no_grad():
    for images, labels in loader:
        images = images.to(DEVICE)

        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probabilities, dim=1)

        actual_label = classes[labels.item()]
        predicted_label = classes[predicted.item()]

        confidence_value = confidence.item() * 100

        image_path = dataset.samples[correct + incorrect][0]
        image_name = Path(image_path).name

        if predicted_label == actual_label:
            correct += 1
            status = "CORRECT"
        else:
            incorrect += 1
            status = "INCORRECT"

        print("\nImage:", image_name)
        print("Actual:", actual_label)
        print("Predicted:", predicted_label)
        print(f"Confidence: {confidence_value:.2f}%")
        print("Status:", status)

print("\n===== SUMMARY =====")
print("Correct predictions:", correct)
print("Incorrect predictions:", incorrect)
print("Total:", len(dataset))
