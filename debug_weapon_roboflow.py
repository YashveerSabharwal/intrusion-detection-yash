"""
Usage:
    python debug_weapon_roboflow.py input/gun.mp4
    python debug_weapon_roboflow.py input/police_none.mp4

Tests WeaponDetectorV2 (multi_model_detector.py, Roboflow-hosted,
weapon-detection-db7n2/2) -- the THIRD WEAPON candidate tried this
project. The first two (WeaponDetector: firearm_yolo, ThreatDetector)
both failed completely on real footage despite higher advertised
benchmarks -- see their class docstrings in multi_model_detector.py.

Zero-shot YOLO-World's own "firearm" prompt is CURRENTLY the only
WEAPON source in build_registry() and is working (0.397 true positive,
no observed false positives). This script exists to check whether
WeaponDetectorV2 is a genuine ADDITIONAL signal worth adding alongside
it, or a third failure to add to the pile -- not to replace the
zero-shot prompt, which stays regardless.

Run against BOTH a true-positive clip (gun.mp4) and a hard-negative
clip with visual clutter (police_none.mp4) -- same as debug_weapon.py.

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

from detector import IncidentDetector
from multi_model_detector import WeaponDetectorV2

video_path = sys.argv[1] if len(sys.argv) > 1 else "input/gun.mp4"

api_key = os.environ.get("ROBOFLOW_API_KEY")
if not api_key:
    print("ERROR: set ROBOFLOW_API_KEY in a .env file or as an environment variable first.")
    sys.exit(1)

cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
print(f"Video: {video_path}")
print(f"{frame_count} frames, {fps:.1f} fps, {frame_count/fps:.1f}s duration")

zero_shot = IncidentDetector()
weapon_v2 = WeaponDetectorV2(api_key=api_key, conf_threshold=0.05)

frame_idx = 0
all_v2_confidences = []
all_zeroshot_confidences = []
raw_response_shown = False

while True:
    ok, frame = cap.read()
    if not ok:
        break
    if frame_idx % 15 == 0:
        if not raw_response_shown:
            raw_result = weapon_v2.client.infer(frame, model_id=weapon_v2.model_id)
            print(f"\nRAW API response (frame {frame_idx}):")
            print(raw_result)
            print()
            raw_response_shown = True

        zs_raw = zero_shot.infer_raw(frame)
        zs_weapon = [d for d in zs_raw if d.category == "WEAPON"]
        v2_dets = weapon_v2.detect(frame)

        print(f"\nframe {frame_idx}:")
        print(f"  zero-shot WEAPON hits:  {[(round(d.confidence, 3), d.prompt) for d in zs_weapon]}")
        print(f"  roboflow_weapon hits:   {[(round(d.confidence, 3), d.prompt) for d in v2_dets]}")

        all_zeroshot_confidences.extend(d.confidence for d in zs_weapon)
        all_v2_confidences.extend(d.confidence for d in v2_dets)
    frame_idx += 1

cap.release()

print("\n" + "=" * 50)
print(f"SUMMARY for {video_path}")
print("=" * 50)
if all_v2_confidences:
    print(f"roboflow_weapon:  min={min(all_v2_confidences):.3f}  "
          f"max={max(all_v2_confidences):.3f}  n={len(all_v2_confidences)}")
else:
    print("roboflow_weapon:  NO detections at all on this clip (at 0.05 floor)")

if all_zeroshot_confidences:
    print(f"zero-shot WEAPON:  min={min(all_zeroshot_confidences):.3f}  "
          f"max={max(all_zeroshot_confidences):.3f}  n={len(all_zeroshot_confidences)}")
else:
    print("zero-shot WEAPON:  NO detections at all on this clip (at raw floor)")

print("\nRun against gun.mp4 (true positive) AND police_none.mp4 (hard")
print("negative), then send both summaries back -- if roboflow_weapon")
print("shows a clean gap the way yolov10fire eventually did for FIRE/SMOKE,")
print("it's worth adding as a second WEAPON source. If it overlaps with")
print("false positives like the first two candidates did, it joins them.")