"""
Usage: python debug_compare.py input/fire_house.mp4

1. Saves a middle frame as debug_frame.jpg so you can just look at it.
2. Runs the OLD zero-shot YOLO-World FIRE/SMOKE prompts on a few sampled
   frames.
3. Runs the NEW Thalos fire/smoke model on the same frames.
Side-by-side, so we know whether NEITHER model sees anything (content
problem) or only one does (model problem).
"""
import sys
import cv2

import config
from detector import IncidentDetector
from multi_model_detector import FireSafetyDetector

video_path = sys.argv[1] if len(sys.argv) > 1 else "input/fire_house.mp4"

cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
print(f"Video: {frame_count} frames, {fps:.1f} fps, {frame_count/fps:.1f}s duration")

# grab and save the middle frame so you can eyeball it
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
ok, mid_frame = cap.read()
if ok:
    cv2.imwrite("debug_frame.jpg", mid_frame)
    print(f"Saved debug_frame.jpg (frame {frame_count // 2}) -- open it and look.")
else:
    print("Could not read a middle frame -- video may be corrupt or very short.")

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

zero_shot = IncidentDetector()
thalos = FireSafetyDetector(conf_threshold=0.05)

frame_idx = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if frame_idx % 15 == 0:
        zs_raw = zero_shot.infer_raw(frame)  # already floored at config.RAW_CONFIDENCE_FLOOR
        zs_fire_smoke = [d for d in zs_raw if d.category in ("FIRE", "SMOKE")]

        th_results = thalos.model(frame, conf=0.05, verbose=False)[0]

        print(f"\nframe {frame_idx}:")
        print(f"  zero-shot FIRE/SMOKE hits: {[(d.category, round(d.confidence,3), d.prompt) for d in zs_fire_smoke]}")
        print(f"  thalos raw box count: {len(th_results.boxes)}")
        for box in th_results.boxes:
            print(f"    thalos: {th_results.names[int(box.cls)]} conf={float(box.conf):.3f}")
    frame_idx += 1

cap.release()