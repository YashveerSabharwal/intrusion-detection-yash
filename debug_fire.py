"""
Run this directly against your fire clip -- bypasses main.py entirely.
Prints the model's ACTUAL class names (to check against
FireSafetyDetector.CLASS_TO_CATEGORY's assumptions) and every raw box
it finds, at a near-zero confidence floor, so you can see if the model
is detecting NOTHING vs detecting something under a different label
than expected.

Usage: python debug_fire.py input/fire_house.mp4
"""
import sys
import cv2
from multi_model_detector import FireSafetyDetector

video_path = sys.argv[1] if len(sys.argv) > 1 else "input/fire_house.mp4"

detector = FireSafetyDetector(conf_threshold=0.05)  # near-zero floor for debugging
print("Model's actual class names:", detector.model.names)
print("Your CLASS_TO_CATEGORY keys:", list(detector.CLASS_TO_CATEGORY.keys()))
print()

cap = cv2.VideoCapture(video_path)
frame_idx = 0
any_box_at_all = False
any_mapped_category = False

while True:
    ok, frame = cap.read()
    if not ok:
        break
    if frame_idx % 15 == 0:  # sample every 15th raw frame, adjust as needed
        results = detector.model(frame, conf=0.05, verbose=False)[0]
        if len(results.boxes) > 0:
            any_box_at_all = True
        for box in results.boxes:
            cls_name = results.names[int(box.cls)]
            conf = float(box.conf)
            mapped = detector.CLASS_TO_CATEGORY.get(cls_name)
            if mapped:
                any_mapped_category = True
            print(f"frame {frame_idx}: class='{cls_name}' conf={conf:.3f} -> mapped to {mapped}")
    frame_idx += 1

cap.release()

print()
if not any_box_at_all:
    print("RESULT: model drew ZERO boxes on ANY sampled frame, at ANY confidence.")
    print("This is a video/content issue, not a class-name mapping issue --")
    print("check the clip actually contains flame-shaped structure, not just")
    print("a solid/overexposed color fill (bounding-box detectors need an")
    print("object shape to draw a box around).")
elif not any_mapped_category:
    print("RESULT: model DID draw boxes, but NONE matched CLASS_TO_CATEGORY.")
    print("This IS the bug -- copy the exact class names printed above into")
    print("FireSafetyDetector.CLASS_TO_CATEGORY in multi_model_detector.py.")
else:
    print("RESULT: at least one box mapped to a real category -- check the")
    print("printed confidence values against CONFIDENCE_THRESHOLDS['FIRE']/['SMOKE']")
    print("in config.py to see if they're just too low to clear threshold.")