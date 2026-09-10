"""
debug_candidate.py -- standalone test harness for any candidate
replacement model, run across ALL labeled clips at once. Reuses the
same EXPECTED ground-truth mapping as batch_test.py so results are
directly comparable to the existing thalos/firearm_yolo numbers.

Why standalone rather than dropping straight into multi_model_detector.py:
if two models covering the same category (e.g. old thalos + new
candidate) both run at once, detector._dedupe() only keeps the
higher-confidence overlapping box per category -- you'd never see
whether the NEW model actually fired, only whichever one "won". This
script isolates ONE candidate model at a time so its real behavior is
visible before it's trusted enough to go into build_registry().

Usage:
    # TommyNgx fire/smoke model (GATED -- needs a HuggingFace token,
    # see instructions below)
    python debug_candidate.py \\
        --repo TommyNgx/YOLOv10-Fire-and-Smoke-Detection \\
        --filename best.pt \\
        --classes fire,smoke \\
        --category-map fire=FIRE,smoke=SMOKE \\
        --token hf_xxx

    # Subh775 multi-class threat model (ungated)
    python debug_candidate.py \\
        --repo Subh775/Threat-Detection-YOLOv8n \\
        --filename weights/best.pt \\
        --category-map gun=WEAPON,grenade=WEAPON

--------------------------------------------------------------------
GETTING A HUGGINGFACE TOKEN (only needed for gated repos like TommyNgx):
  1. Create a free account at https://huggingface.co if you don't have one.
  2. Open the model page (e.g.
     https://huggingface.co/TommyNgx/YOLOv10-Fire-and-Smoke-Detection)
     and click "Agree and access repository".
  3. Go to https://huggingface.co/settings/tokens, create a new token
     (read access is enough).
  4. Either pass it with --token, or set it once for your shell:
       PowerShell:  $env:HF_TOKEN="hf_xxxxxxxxxxxx"
     then omit --token and this script will pick it up automatically.
--------------------------------------------------------------------
"""
import argparse
import os
from pathlib import Path

import cv2

# Reuse the exact same ground-truth mapping as batch_test.py so numbers
# from a candidate model are directly comparable to what thalos/
# firearm_yolo scored on the same clips.
from batch_test import EXPECTED, INPUT_DIR


def parse_category_map(spec):
    """'fire=FIRE,smoke=SMOKE' -> {'fire': 'FIRE', 'smoke': 'SMOKE'}, keys lowercased."""
    out = {}
    for pair in spec.split(","):
        cls_name, category = pair.split("=")
        out[cls_name.strip().lower()] = category.strip()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="HuggingFace repo id, e.g. TommyNgx/YOLOv10-Fire-and-Smoke-Detection")
    parser.add_argument("--filename", required=True, help="Weights filename inside the repo, e.g. best.pt")
    parser.add_argument("--category-map", required=True,
                         help="Comma-separated model_class_name=ALERT_CATEGORY pairs, e.g. 'fire=FIRE,smoke=SMOKE'. "
                              "Matched as a case-insensitive SUBSTRING against the model's real class names.")
    parser.add_argument("--token", default=None, help="HuggingFace token, for gated repos. Falls back to HF_TOKEN env var.")
    parser.add_argument("--conf-floor", type=float, default=0.05, help="Raw confidence floor for this test.")
    parser.add_argument("--sample-every", type=int, default=15, help="Sample every Nth raw frame.")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    category_map = parse_category_map(args.category_map)

    print(f"Downloading {args.repo} / {args.filename} ...")
    from huggingface_hub import hf_hub_download
    try:
        weights_path = hf_hub_download(repo_id=args.repo, filename=args.filename, token=token)
    except Exception as e:
        print(f"\nDOWNLOAD FAILED: {e}")
        print("If this is a gated repo (401/403 error), check:")
        print("  1. You clicked 'Agree and access repository' on the model's HF page while logged in.")
        print("  2. Your token has at least read access.")
        print("  3. You passed --token or set $env:HF_TOKEN correctly.")
        return

    from ultralytics import YOLO
    model = YOLO(weights_path)
    print(f"\nModel's actual class names: {model.names}")
    print(f"Your category map (lowercased keys): {category_map}")
    unmatched_classes = [n for n in model.names.values() if not any(k in n.lower() for k in category_map)]
    if unmatched_classes:
        print(f"NOTE: these model classes don't match any --category-map entry and will be IGNORED: {unmatched_classes}")
    print()

    overall_by_category = {}  # category -> {"true_pos": [...], "false_pos": [...]}

    for clip_name, expected_categories in EXPECTED.items():
        clip_path = INPUT_DIR / clip_name
        if not clip_path.exists():
            print(f"SKIP {clip_name}: file not found")
            continue

        cap = cv2.VideoCapture(str(clip_path))
        frame_idx = 0
        clip_hits = {}  # category -> list of confidences

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % args.sample_every == 0:
                results = model(frame, conf=args.conf_floor, verbose=False)[0]
                for box in results.boxes:
                    cls_name = results.names[int(box.cls)].lower()
                    category = None
                    for key, cat in category_map.items():
                        if key in cls_name:
                            category = cat
                            break
                    if category is None:
                        continue
                    clip_hits.setdefault(category, []).append(float(box.conf))
            frame_idx += 1
        cap.release()

        print(f"{clip_name}  (expected: {expected_categories or 'none'})")
        if not clip_hits:
            print("    (no matching detections at all)")
        for category, confs in clip_hits.items():
            is_expected = category in expected_categories
            tag = "TRUE POSITIVE" if is_expected else "FALSE POSITIVE"
            print(f"    {category}: min={min(confs):.3f} max={max(confs):.3f} n={len(confs)}  <-- {tag}")

            bucket = overall_by_category.setdefault(category, {"true_pos": [], "false_pos": []})
            key = "true_pos" if is_expected else "false_pos"
            bucket[key].extend(confs)
        print()

    print("=" * 70)
    print(f"SUMMARY for {args.repo}")
    print("=" * 70)
    for category, bucket in overall_by_category.items():
        tp, fp = bucket["true_pos"], bucket["false_pos"]
        print(f"\n{category}:")
        if tp:
            print(f"  true positives:  min={min(tp):.3f} max={max(tp):.3f} n={len(tp)}")
        else:
            print("  true positives:  NONE OBSERVED")
        if fp:
            print(f"  false positives: min={min(fp):.3f} max={max(fp):.3f} n={len(fp)}")
        else:
            print("  false positives: none observed in this test set")

        if tp and fp and max(fp) >= min(tp):
            print("  VERDICT: ranges OVERLAP -- no threshold can cleanly separate these, same failure as thalos:fire.")
        elif tp and not fp:
            print("  VERDICT: clean true positives, no false positives seen -- promising, but small sample.")
        elif tp and fp:
            print(f"  VERDICT: separable -- a threshold between {max(fp):.3f} and {min(tp):.3f} looks workable.")
        else:
            print("  VERDICT: insufficient data either way.")

    print("\nSend me this full output before we decide whether to wire this")
    print("model into multi_model_detector.py for real.")


if __name__ == "__main__":
    main()