"""
Usage:
    python debug_electrical.py input/electric_shock.mp4
    python debug_electrical.py input/some_hard_negative_clip.mp4

Tests ElectricalArcDetector (multi_model_detector.py), now wired to
a-a-fbw8p/fire-smoke-and-spark-detection/2 (Roboflow-hosted).

config.CONFIDENCE_THRESHOLDS["ELECTRICAL"] = 0.30 has existed since
before this model was wired in -- it was a placeholder ("currently
unreachable/irrelevant until one is wired up"), NOT derived from real
footage. Treat it exactly like the other unvalidated thresholds: run
this against electric_shock.mp4 (the one clip you have that's
plausibly a true positive) and a hard negative, then send both
summaries back.

Requires ROBOFLOW_API_KEY (.env file or exported env var -- see
debug_accident.py's docstring for setup).
"""
import os
import sys
import cv2

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from multi_model_detector import ElectricalArcDetector

video_path = sys.argv[1] if len(sys.argv) > 1 else "input/electric_shock.mp4"

api_key = os.environ.get("ROBOFLOW_API_KEY")
if not api_key:
    print("ERROR: set ROBOFLOW_API_KEY in a .env file or as an environment variable first.")
    sys.exit(1)

cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
print(f"Video: {video_path}")
print(f"{frame_count} frames, {fps:.1f} fps, {frame_count/fps:.1f}s duration")

electrical_model = ElectricalArcDetector(api_key=api_key, conf_threshold=0.05)

frame_idx = 0
all_confidences = []
raw_response_shown = False

while True:
    ok, frame = cap.read()
    if not ok:
        break
    if frame_idx % 15 == 0:
        if not raw_response_shown:
            raw_result = electrical_model.client.infer(frame, model_id=electrical_model.model_id)
            print(f"\nRAW API response (frame {frame_idx}):")
            print(raw_result)
            print()
            raw_response_shown = True

        dets = electrical_model.detect(frame)
        print(f"frame {frame_idx}: {[(round(d.confidence, 3), d.prompt) for d in dets]}")
        all_confidences.extend(d.confidence for d in dets)
    frame_idx += 1

cap.release()

print("\n" + "=" * 50)
print(f"SUMMARY for {video_path}")
print("=" * 50)
if all_confidences:
    print(f"roboflow_arc:  min={min(all_confidences):.3f}  "
          f"max={max(all_confidences):.3f}  "
          f"n={len(all_confidences)}")
else:
    print("roboflow_arc:  NO detections at all on this clip (at 0.05 floor)")

print("\nRun against electric_shock.mp4 (expect true positives, if this")
print("clip actually shows sparking) AND a hard-negative clip, then send")
print("both summaries back to set a real CONFIDENCE_THRESHOLDS['ELECTRICAL']")
print("instead of the current unvalidated 0.30 placeholder.")