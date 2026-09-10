"""
alerts.py
Turns confirmed temporal states into cooldown-limited alerts and writes
them to output/alerts.json.
"""

import json
import time

import config


def frame_to_timestamp(frame_number, fps):
    total_seconds = frame_number / fps if fps > 0 else 0
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class AlertEngine:
    def __init__(self, cooldown_seconds=None):
        self.cooldown_seconds = cooldown_seconds or config.ALERT_COOLDOWN_SECONDS
        self.last_alert_time = {}  # category -> last alert wall-clock-equivalent (video seconds)
        self.alerts = []

    def maybe_alert(self, category, detection, frame_number, video_fps):
        """
        Given a confirmed category and its representative detection,
        create an alert if we're outside the cooldown window for that
        category. Returns the alert dict if one was created, else None.
        """
        video_seconds = frame_number / video_fps if video_fps > 0 else 0
        last = self.last_alert_time.get(category)

        if last is not None and (video_seconds - last) < self.cooldown_seconds:
            return None

        alert = {
            "timestamp": frame_to_timestamp(frame_number, video_fps),
            "event": category,
            "confidence": round(float(detection.confidence), 3),
            "bbox": [int(v) for v in detection.bbox],
            "frame": int(frame_number),
            # NEW: which zero-shot prompt or which dedicated model produced
            # this detection, e.g. "person holding a rifle", "firearm_yolo:gun",
            # "thalos:fire". Without this there was no way to tell, after the
            # fact, which of WEAPON's two sources (zero-shot vs firearm_yolo)
            # triggered a given alert -- needed this to diagnose false positives.
            "source": detection.prompt,
        }
        self.alerts.append(alert)
        self.last_alert_time[category] = video_seconds
        return alert

    def save(self, path=None):
        path = path or config.ALERTS_PATH
        with open(path, "w") as f:
            json.dump(self.alerts, f, indent=2)
        return path