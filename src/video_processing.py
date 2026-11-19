
import cv2
from pathlib import Path


def get_video_info(video_path):
    """Read video metadata."""

    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration = frame_count / fps if fps and fps > 0 else 0

    info = {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_seconds": duration,
    }

    cap.release()
    return info


def extract_frames(video_path, output_dir, interval_seconds=1):
    """
    Extract frames from a video at regular time intervals.

    Example:
        interval_seconds=1  -> 1 frame every second
        interval_seconds=2  -> 1 frame every two seconds
    """

    video_path = str(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        cap.release()
        raise ValueError("Invalid video FPS.")

    frame_step = max(1, round(fps * interval_seconds))

    saved_frames = []
    frame_index = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        if frame_index % frame_step == 0:
            filename = f"frame_{frame_index:06d}.jpg"
            frame_path = output_dir / filename

            saved = cv2.imwrite(str(frame_path), frame)

            if saved:
                saved_frames.append(str(frame_path))

        frame_index += 1

    cap.release()

    return saved_frames


def process_video(video_path, output_dir, interval_seconds=1):
    """Complete video processing pipeline."""

    info = get_video_info(video_path)

    frames = extract_frames(
        video_path=video_path,
        output_dir=output_dir,
        interval_seconds=interval_seconds,
    )

    return {
        "video_info": info,
        "frames_extracted": len(frames),
        "frame_paths": frames,
    }
