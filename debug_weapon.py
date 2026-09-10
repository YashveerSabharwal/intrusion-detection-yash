"""
Usage:
    python debug_weapon.py input/gun.mp4
    python debug_weapon.py input/police_none.mp4

Same idea as debug_compare.py, but for WEAPON instead of FIRE/SMOKE.

WEAPON currently has two sources sharing ONE threshold
(config.CONFIDENCE_THRESHOLDS["WEAPON"] = 0.25):
  1. zero-shot YOLO-World prompts (gun/rifle/knife/etc, config.py)
  2. the dedicated firearm_yolo model (Subh775/Firearm_Detection_Yolov8n,
     multi_model_detector.py) -- guns only, no knives

That's the exact situation FIRE/SMOKE were in before their own
SOURCE_CONFIDENCE_THRESHOLDS fix (section 13) -- don't guess a number,
derive it from real footage the same way.

Run this against BOTH:
  - a clear true-positive clip (gun.mp4)
  - a hard-negative clip with NO weapon but similar visual clutter
    (police_none.mp4 looks like exactly this kind of clip)

Then send me the printed confidence ranges for firearm_yolo on each, and
I'll set SOURCE_CONFIDENCE_THRESHOLDS["firearm_yolo:gun"] to sit between
the true-positive floor and the false-positive ceiling, same as
"thalos:fire"/"thalos:smoke" were set.
"""
import sys
import cv2

from detector import IncidentDetector
from multi_model_detector import WeaponDetector

video_path = sys.argv[1] if len(sys.argv) > 1 else "input/gun.mp4"

cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
print(f"Video: {video_path}")
print(f"{frame_count} frames, {fps:.1f} fps, {frame_count/fps:.1f}s duration")

zero_shot = IncidentDetector()
firearm_model = WeaponDetector(conf_threshold=0.05)

frame_idx = 0
all_firearm_confidences = []
all_zeroshot_weapon_confidences = []

while True:
    ok, frame = cap.read()
    if not ok:
        break
    if frame_idx % 15 == 0:
        zs_raw = zero_shot.infer_raw(frame)  # already floored at config.RAW_CONFIDENCE_FLOOR
        zs_weapon = [d for d in zs_raw if d.category == "WEAPON"]

        firearm_dets = firearm_model.detect(frame)  # already floored at 0.05

        print(f"\nframe {frame_idx}:")
        print(f"  zero-shot WEAPON hits: {[(round(d.confidence,3), d.prompt) for d in zs_weapon]}")
        print(f"  firearm_yolo hits:     {[round(d.confidence,3) for d in firearm_dets]}")

        all_zeroshot_weapon_confidences.extend(d.confidence for d in zs_weapon)
        all_firearm_confidences.extend(d.confidence for d in firearm_dets)

    frame_idx += 1

cap.release()

print("\n" + "=" * 50)
print(f"SUMMARY for {video_path}")
print("=" * 50)
if all_firearm_confidences:
    print(f"firearm_yolo:  min={min(all_firearm_confidences):.3f}  "
          f"max={max(all_firearm_confidences):.3f}  "
          f"n={len(all_firearm_confidences)}")
else:
    print("firearm_yolo:  NO detections at all on this clip (at 0.05 floor)")

if all_zeroshot_weapon_confidences:
    print(f"zero-shot WEAPON:  min={min(all_zeroshot_weapon_confidences):.3f}  "
          f"max={max(all_zeroshot_weapon_confidences):.3f}  "
          f"n={len(all_zeroshot_weapon_confidences)}")
else:
    print("zero-shot WEAPON:  NO detections at all on this clip (at raw floor)")

print("\nRun this against gun.mp4 (expect firearm_yolo true positives) AND")
print("police_none.mp4 (expect firearm_yolo false positives, if any) then")
print("send both summaries back -- that's what's needed to set a real")
print("SOURCE_CONFIDENCE_THRESHOLDS['firearm_yolo:gun'] value.")