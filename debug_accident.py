"""
Usage:
    python debug_accident.py input/car_accident.mp4
    python debug_accident.py input/car_fire_accident.mp4
    python debug_accident.py input/some_hard_negative_clip.mp4

Same idea as debug_weapon.py, but for the new ACCIDENT category
(road-accident-qqade, Roboflow-hosted, wired into multi_model_detector.py
as AccidentDetector this session).

ACCIDENT currently has ONE source (Roboflow's "Accident" class) and NO
config.SOURCE_CONFIDENCE_THRESHOLDS entry yet -- config.CONFIDENCE_THRESHOLDS
["ACCIDENT"] = 0.40 is an unvalidated starting guess, not derived from
real footage.

Run this against BOTH:
  - a clear true-positive clip (car_accident.mp4, car_fire_accident.mp4)
  - a hard-negative clip with NO accident but similar visual clutter
    (e.g. general traffic footage, or any clip you know has no crash)

Then send me the printed confidence ranges and I'll set
CONFIDENCE_THRESHOLDS["ACCIDENT"] (or add a
SOURCE_CONFIDENCE_THRESHOLDS["roboflow_accident:Accident"] entry if a
second ACCIDENT source ever gets added) to sit between the true-positive
floor and the false-positive ceiling -- same derivation method as
fire/smoke and fight, not a guess.

Requires ROBOFLOW_API_KEY. Two ways to provide it:
  1. A .env file in the project root (see .env.example) -- loaded
     automatically below if python-dotenv is installed
     (pip install python-dotenv).
  2. An exported environment variable in your current terminal session.

The free Roboflow "Public" plan API key works for this specific model
since road-accident-qqade is a public project.
"""
import os
import sys
import cv2

try:
    from dotenv import load_dotenv
    load_dotenv()  # reads .env in the current directory, if present
except ImportError:
    pass  # fine if python-dotenv isn't installed -- falls back to a real env var

from multi_model_detector import AccidentDetector

video_path = sys.argv[1] if len(sys.argv) > 1 else "input/car_accident.mp4"

api_key = os.environ.get("ROBOFLOW_API_KEY")
if not api_key:
    print("ERROR: set ROBOFLOW_API_KEY as an environment variable first, e.g.")
    print("  export ROBOFLOW_API_KEY=your_key_here")
    sys.exit(1)

cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
print(f"Video: {video_path}")
print(f"{frame_count} frames, {fps:.1f} fps, {frame_count/fps:.1f}s duration")

accident_model = AccidentDetector(api_key=api_key, conf_threshold=0.05)  # near-zero floor for debugging

frame_idx = 0
all_confidences = []
raw_response_shown = False

while True:
    ok, frame = cap.read()
    if not ok:
        break
    if frame_idx % 15 == 0:  # sample every 15th raw frame, adjust as needed
        # Print the RAW API response for the first sampled frame only --
        # this shows us exactly what Roboflow sent back (an "error" key?
        # a "predictions": [] with something else useful? a totally
        # different shape than expected?) instead of the already-filtered
        # Detection objects, which hide that information.
        if not raw_response_shown:
            raw_result = accident_model.client.infer(frame, model_id=accident_model.model_id)
            print(f"\nRAW API response (frame {frame_idx}):")
            print(raw_result)
            print()
            raw_response_shown = True

        dets = accident_model.detect(frame)  # already floored at 0.05
        print(f"frame {frame_idx}: {[(round(d.confidence, 3), d.prompt) for d in dets]}")
        all_confidences.extend(d.confidence for d in dets)
    frame_idx += 1

cap.release()

print("\n" + "=" * 50)
print(f"SUMMARY for {video_path}")
print("=" * 50)
if all_confidences:
    print(f"roboflow_accident:  min={min(all_confidences):.3f}  "
          f"max={max(all_confidences):.3f}  "
          f"n={len(all_confidences)}")
else:
    print("roboflow_accident:  NO detections at all on this clip (at 0.05 floor)")

print("\nRun this against car_accident.mp4 / car_fire_accident.mp4 (expect")
print("true positives) AND a hard-negative clip with no accident (expect")
print("no detections, or clearly lower confidence) then send both")
print("summaries back -- that's what's needed to set a real")
print("CONFIDENCE_THRESHOLDS['ACCIDENT'] value instead of the current")
print("unvalidated 0.40 guess.")