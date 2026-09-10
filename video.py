"""
video.py
OpenCV-based video reading/writing and simple overlay drawing.
"""

import os

import cv2

import config


COLORS = {
    "FIRE": (0, 69, 255),           # orange-red (BGR)
    "SMOKE": (180, 180, 180),       # grey
    "WEAPON": (0, 0, 255),          # red
    "FALLEN_PERSON": (0, 255, 255), # yellow
    "FALLEN_TREE": (19, 69, 139),   # brown
    "FIGHT": (0, 140, 255),         # dark orange -- distinct from FIRE's orange-red
    "ELECTRICAL": (255, 255, 0),    # cyan -- distinct from FIRE/WEAPON's red family
}


class VideoSource:
    def __init__(self, path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Input video not found: {path}")

        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video (unsupported codec or corrupt file): {path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.width == 0 or self.height == 0:
            raise RuntimeError(f"Video reports zero dimensions, likely corrupt: {path}")

    def frames(self, max_frames=None):
        """Yield (frame_number, frame) for every raw frame in the video."""
        idx = 0
        while True:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                break
            yield idx, frame
            idx += 1
            if max_frames is not None and idx >= max_frames:
                break

    def release(self):
        self.cap.release()


class VideoSink:
    def __init__(self, path, fps, width, height):
        out_dir = os.path.dirname(path)
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if not self.writer.isOpened():
            raise RuntimeError(f"Could not open video writer for: {path}")

    def write(self, frame):
        self.writer.write(frame)

    def release(self):
        self.writer.release()


def draw_detections(frame, detections, confirmed_categories):
    """Draw bounding boxes + label + confidence for each detection.
    Categories that are temporally confirmed get an extra alert banner."""
    for det in detections:
        color = COLORS.get(det.category, (255, 255, 255))
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{det.category} {det.confidence:.2f}"
        cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    y_offset = 30
    for category, is_confirmed in confirmed_categories.items():
        if not is_confirmed:
            continue
        color = COLORS.get(category, (0, 0, 255))
        cv2.putText(frame, f"!!! {category} ALERT !!!", (15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        y_offset += 30

    return frame