"""
batch_test.py -- run main.py against every clip in your labeled test set
and print ONE consolidated summary table instead of debugging one
screenshot at a time.

EXPECTED below is a guess based on filenames -- edit it if I got any of
these wrong (e.g. if car_accident.mp4 is meant to test something
specific, or if electric_shock.mp4 is meant to be a hard negative
rather than a true positive).

Usage:
    python batch_test.py
    python batch_test.py --max-frames 100    # faster, partial-clip pass
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import config

# clip filename (relative to input/) -> categories that SHOULD fire.
# Empty list = hard negative, expect NO alerts at all.
EXPECTED = {
    "fire_house.mp4":         ["FIRE", "SMOKE"],
    "forest_fire.mp4":        ["FIRE", "SMOKE"],
    "car_fire_accident.mp4":  ["FIRE", "SMOKE", "ACCIDENT"],  # ACCIDENT added this session -- see debug_accident.py before trusting this
    "gun.mp4":                ["WEAPON"],
    "police_none.mp4":        [],                   # hard negative -- WEAPON false-positive check
    "real_fight.mp4":         ["FIGHT"],
    "dog_attack.mp4":         ["FIGHT","FALLEN_PERSON"],
    "electric_shock.mp4":     ["ELECTRICAL"],        # known unresolved -- no detector configured yet
    "car_accident.mp4":       ["ACCIDENT"],          # ACCIDENT wired in this session (road-accident-qqade) -- unvalidated threshold, run debug_accident.py first

    # --- added this session, from input/ folder listing ---
    "fire2.mp4":               ["FIRE", "SMOKE"],       # generic fire clip, name suggests confident guess
    "fire3.mp4":                ["FIRE", "SMOKE"],       # same
    "fire_people.mp4":          ["FIRE"],                # people near/around fire -- FIRE expected, SMOKE uncertain, verify visually
    "weapon.mp4":                ["WEAPON"],
    "reddit_fight.mp4":          ["FIGHT","FALLEN_PERSON"],
    "normal.mp4":                [],                     # hard negative -- should trigger NOTHING; good general false-positive check across all categories

    # UNCERTAIN -- filename alone doesn't tell me enough, please correct:
    "clash.mp4":                 ["FIGHT"],   # guessing this is a physical altercation/clash -- confirm or fix
    "anaar.mp4":                 [],           # "anaar" = a firework/sparkler type -- treating as HARD NEGATIVE since it's not real fire, but flames-from-firework may false-positive FIRE_PROMPTS ("fireball", "explosion") -- worth watching closely
    "diwali.mp4":                [],           # festival lights/fireworks -- same false-positive risk as anaar.mp4, hard negative for now
    "fuljhadi.mp4":              [],           # "fuljhadi" = sparkler -- same reasoning as anaar/diwali
}

INPUT_DIR = Path("input")
RESULTS_DIR = Path("output/batch_results")


def run_one(clip_name, max_frames=None):
    clip_path = INPUT_DIR / clip_name
    if not clip_path.exists():
        return None, f"MISSING FILE: {clip_path}"

    cmd = [sys.executable, "main.py", "--input", str(clip_path)]
    if max_frames:
        cmd += ["--max-frames", str(max_frames)]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # main.py's own error handling (missing input, model load failure,
        # etc.) prints via print() -- i.e. stdout, not stderr -- then calls
        # sys.exit(1). Only showing stderr here was hiding those messages
        # entirely, making every real failure look identical/blank.
        tail_out = result.stdout[-800:] if result.stdout else "(empty)"
        tail_err = result.stderr[-800:] if result.stderr else "(empty)"
        return None, (f"main.py FAILED (exit {result.returncode}):\n"
                       f"--- stdout ---\n{tail_out}\n"
                       f"--- stderr ---\n{tail_err}")

    alerts_path = Path(config.ALERTS_PATH)
    if not alerts_path.exists():
        return None, "main.py ran but produced no alerts.json"

    with open(alerts_path) as f:
        alerts = json.load(f)

    # Archive this clip's alerts.json + report.html so they don't get
    # overwritten by the next clip in the loop.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = clip_path.stem
    shutil.copy(alerts_path, RESULTS_DIR / f"{stem}_alerts.json")
    report_path = Path(config.REPORT_PATH)
    if report_path.exists():
        shutil.copy(report_path, RESULTS_DIR / f"{stem}_report.html")

    return alerts, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-frames", type=int, default=None,
                         help="Limit frames per clip for a faster pass.")
    args = parser.parse_args()

    rows = []
    for clip_name, expected_categories in EXPECTED.items():
        print(f"\n{'='*60}\nRunning {clip_name} ...")
        alerts, error = run_one(clip_name, max_frames=args.max_frames)

        if error:
            rows.append((clip_name, expected_categories, None, {}, error))
            print(f"  ERROR: {error}")
            continue

        # Group by category and show confidence range + source, so we can
        # actually see WHY something fired instead of just THAT it fired.
        by_category = {}
        for a in alerts:
            by_category.setdefault(a["event"], []).append(a)

        fired_categories = sorted(by_category.keys())
        fired_detail = {}
        for cat, cat_alerts in by_category.items():
            confs = [a["confidence"] for a in cat_alerts]
            sources = sorted(set(a.get("source", "?") for a in cat_alerts))
            fired_detail[cat] = f"{min(confs):.3f}-{max(confs):.3f} via {sources}"

        rows.append((clip_name, expected_categories, fired_categories, fired_detail, None))
        print(f"  Fired: {fired_categories or '(none)'}")
        for cat, detail in fired_detail.items():
            print(f"    {cat}: {detail}")

    # ------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------
    print("\n\n" + "=" * 100)
    print("DETAIL (confidence range + which model/prompt fired), for false positives especially:")
    print("=" * 100)

    for clip_name, expected, fired, fired_detail, error in rows:
        if error:
            print(f"\n{clip_name}: ERROR (see above)")
            continue

        expected_set = set(expected)
        fired_set = set(fired)
        false_positives = fired_set - expected_set
        missed = expected_set - fired_set

        print(f"\n{clip_name}  (expected: {expected or 'none'})")
        for cat in fired:
            flag = " <-- FALSE POSITIVE" if cat in false_positives else ""
            print(f"    {cat}: {fired_detail[cat]}{flag}")
        if missed:
            print(f"    MISSED: {sorted(missed)}")

    print("\n" + "=" * 100)
    print(f"{'CLIP':<25} {'EXPECTED':<25} {'FIRED':<25} {'VERDICT'}")
    print("=" * 100)

    for clip_name, expected, fired, fired_detail, error in rows:
        if error:
            print(f"{clip_name:<25} {'-':<25} {'-':<25} ERROR (see above)")
            continue

        expected_set = set(expected)
        fired_set = set(fired) if fired else set()

        missed = expected_set - fired_set
        false_positives = fired_set - expected_set

        if not missed and not false_positives:
            verdict = "PASS"
        else:
            parts = []
            if missed:
                parts.append(f"MISSED: {sorted(missed)}")
            if false_positives:
                parts.append(f"FALSE POSITIVE: {sorted(false_positives)}")
            verdict = " | ".join(parts)

        print(f"{clip_name:<25} {str(expected):<25} {str(sorted(fired) if fired else []):<25} {verdict}")

    print("=" * 100)
    print(f"\nPer-clip alerts.json/report.html archived under {RESULTS_DIR}/")
    print("Send me this whole summary table (copy-paste the terminal output)")
    print("plus any report.html for a clip marked FALSE POSITIVE or MISSED,")
    print("and we fix categories systematically instead of one clip at a time.")


if __name__ == "__main__":
    main()