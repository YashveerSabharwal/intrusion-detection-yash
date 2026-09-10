"""
Usage:
    python debug_fight_roboflow.py input/real_fight.mp4
    python debug_fight_roboflow.py input/police_none.mp4
    python debug_fight_roboflow.py input/reddit_fight.mp4

Tests FightDetectorV2 (multi_model_detector.py, Roboflow-hosted,
fight-detection-7xdy7/2, YOLOv11 keypoint model) SIDE BY SIDE with the
currently-ACTIVE local FightDetector (Musawer14/fight_detection_yolov8),
which already has one confirmed true positive at threshold 0.60.

This is a COMPARISON, not a replacement test -- FightDetectorV2 only
takes over if it's demonstrably better on real footage, not just on
paper (its mAP@50 83.7% vs the active model's untested-but-working
threshold don't tell you which wins on YOUR clips).

Requires ROBOFLOW_API_KEY (.env file or exported env var).
"""
import os
import sys
import cv2

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from multi_model_detector import FightDetector, FightDetectorV2

video_path = sys.argv[1] if len(sys.argv) > 1 else "input/real_fight.mp4"

api_key = os.environ.get("ROBOFLOW_API_KEY")
if not api_key:
    print("ERROR: set ROBOFLOW_API_KEY in a .env file or as an environment variable first.")
    sys.exit(1)

cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
print(f"Video: {video_path}")
print(f"{frame_count} frames, {fps:.1f} fps, {frame_count/fps:.1f}s duration")

active_model = FightDetector(conf_threshold=0.05)
candidate_model = FightDetectorV2(api_key=api_key, conf_threshold=0.05)

frame_idx = 0
all_active_confidences = []
all_candidate_confidences = []
raw_response_shown = False

while True:
    ok, frame = cap.read()
    if not ok:
        break
    if frame_idx % 15 == 0:
        if not raw_response_shown:
            raw_result = candidate_model.client.infer(frame, model_id=candidate_model.model_id)
            print(f"\nRAW API response (frame {frame_idx}):")
            print(raw_result)
            print()
            raw_response_shown = True

        active_dets = active_model.detect(frame)
        candidate_dets = candidate_model.detect(frame)

        print(f"\nframe {frame_idx}:")
        print(f"  ACTIVE (fight_yolo) hits:     {[(round(d.confidence, 3), d.prompt) for d in active_dets]}")
        print(f"  CANDIDATE (roboflow) hits:    {[(round(d.confidence, 3), d.prompt) for d in candidate_dets]}")

        all_active_confidences.extend(d.confidence for d in active_dets)
        all_candidate_confidences.extend(d.confidence for d in candidate_dets)
    frame_idx += 1

cap.release()

print("\n" + "=" * 50)
print(f"SUMMARY for {video_path}")
print("=" * 50)
if all_active_confidences:
    print(f"ACTIVE fight_yolo:      min={min(all_active_confidences):.3f}  "
          f"max={max(all_active_confidences):.3f}  n={len(all_active_confidences)}")
else:
    print("ACTIVE fight_yolo:      NO detections at all on this clip")

if all_candidate_confidences:
    print(f"CANDIDATE roboflow:     min={min(all_candidate_confidences):.3f}  "
          f"max={max(all_candidate_confidences):.3f}  n={len(all_candidate_confidences)}")
else:
    print("CANDIDATE roboflow:     NO detections at all on this clip")

print("\nRun against real_fight.mp4 / reddit_fight.mp4 (true positives) AND")
print("police_none.mp4 / dog_attack.mp4 (hard negatives), then send all")
print("summaries back. The candidate only replaces the active model if it")
print("shows a cleaner true/false-positive gap on YOUR footage.")