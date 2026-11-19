
import cv2
from pathlib import Path


def load_face_detector():
    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    detector = cv2.CascadeClassifier(cascade_path)

    if detector.empty():
        raise RuntimeError("Failed to load face detector.")

    return detector


def detect_faces(frame, detector=None):
    """Return bounding boxes of faces detected in an image."""

    if detector is None:
        detector = load_face_detector()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    return faces


def crop_faces(frame, faces, padding=0.20):
    """Crop faces with a small surrounding margin."""

    height, width = frame.shape[:2]
    crops = []

    for (x, y, w, h) in faces:

        pad_x = int(w * padding)
        pad_y = int(h * padding)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(width, x + w + pad_x)
        y2 = min(height, y + h + pad_y)

        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size > 0:
            crops.append(face_crop)

    return crops


def process_frames(frame_dir, output_dir):
    """Detect faces in extracted frames and save cropped faces."""

    frame_dir = Path(frame_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    detector = load_face_detector()

    frame_files = sorted(frame_dir.glob("*.jpg"))

    total_faces = 0
    frames_with_faces = 0
    frames_without_faces = 0

    for frame_path in frame_files:

        frame = cv2.imread(str(frame_path))

        if frame is None:
            print(f"Skipping unreadable image: {frame_path.name}")
            continue

        faces = detect_faces(frame, detector)

        if len(faces) == 0:
            frames_without_faces += 1
            continue

        frames_with_faces += 1

        crops = crop_faces(frame, faces)

        for index, crop in enumerate(crops):

            filename = (
                f"{frame_path.stem}"
                f"_face_{index:02d}.jpg"
            )

            save_path = output_dir / filename

            cv2.imwrite(str(save_path), crop)
            total_faces += 1

    return {
        "total_frames": len(frame_files),
        "frames_with_faces": frames_with_faces,
        "frames_without_faces": frames_without_faces,
        "total_faces_detected": total_faces,
        "output_dir": str(output_dir)
    }
