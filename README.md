# CCTV Incident Detector (V1 -- Zero-Shot Prototype)

## 1. What this does

Watches a CCTV MP4 and flags five incident categories:

- FIRE
- SMOKE
- WEAPON
- FALLEN_PERSON
- FALLEN_TREE

Output: an annotated MP4 (`output/annotated.mp4`) and a JSON list of
confirmed alerts (`output/alerts.json`).

## 2. Why YOLO-World

YOLO-World is an open-vocabulary detector: instead of being limited to
80 fixed COCO classes, you give it text prompts ("fire", "person lying
on the ground", etc.) and it detects those directly, with no training.
That's what makes a V1 possible without collecting or labeling any data.

## 3. Hardware assumptions

- CPU only, no CUDA, no NVIDIA GPU (tested against a Lenovo ThinkPad).
- Default 2 FPS inference rate -- CPU can't do real-time 25-30 FPS
  open-vocabulary detection, and this project doesn't try to.

## 4. Installation (Windows, CPU-only)

```powershell
cd cctv_detector
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

This installs the CPU build of torch/torchvision by default on Windows
(no CUDA packages are pulled in). If `pip` ever tries to grab a CUDA
build, install the CPU wheel explicitly instead:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## 5. Model setup

No manual download needed in the normal case. On first run, Ultralytics
automatically downloads `yolov8s-worldv2.pt` (the checkpoint set in
`config.py`) into the working directory.

If your environment has no internet access at runtime, download the
checkpoint ahead of time on a machine that does, then drop the `.pt`
file into the project root before running.

**Note:** the exact YOLO-World class-setting API (`model.set_classes(...)`)
can shift slightly between Ultralytics versions. `detector.py` calls it
in the current documented way, but if you see an error there, run
`pip show ultralytics` and check the installed version's YOLO-World docs
-- the fix is a one-line change in `detector.py`, `_load_model()`.

## 6. Running it

```powershell
python main.py --input input/test.mp4
```

Custom output path:

```powershell
python main.py --input input/test.mp4 --output output/result.mp4
```

Quick test on the first 100 frames only:

```powershell
python main.py --input input/test.mp4 --max-frames 100
```

Show a live preview window while processing:

```powershell
python main.py --input input/test.mp4 --display
```

## 7. Changing prompts

All prompts live in `config.py` under `FIRE_PROMPTS`, `SMOKE_PROMPTS`,
`WEAPON_PROMPTS`, `FALLEN_PERSON_PROMPTS`, `FALLEN_TREE_PROMPTS`. Add,
remove, or reword any prompt -- e.g. change `"gun"` to `"a handgun"` --
without touching any other file. Every prompt must map to exactly one
of the 5 categories; `detector.py` builds that mapping automatically
from `config.py`.

## 8. Changing confidence thresholds

Edit `CONFIDENCE_THRESHOLDS` in `config.py`, one value per category
(0-1 scale). These starting values are guesses, not tuned numbers --
raise a category's threshold if it's too trigger-happy, lower it if
it's missing real events.

## 9. Temporal filtering

A single detected frame never creates an alert. `temporal.py` keeps a
sliding window (`TEMPORAL_WINDOW`, default 5 sampled frames) per
category and only "confirms" a category once it's hit at least
`MIN_HITS[category]` times inside that window. This is the main
defense against one-off false positives (a flash of orange light, a
momentary misread, etc).

Once confirmed, `ALERT_COOLDOWN_SECONDS` (default 20) prevents the same
category from spamming new alert entries every couple seconds while
the same ongoing event is still visible -- the on-screen banner keeps
showing, but `alerts.json` only gets one new entry per cooldown window.

## 10. Output format

`output/alerts.json`:

```json
[
  {
    "timestamp": "00:01:32",
    "event": "FIRE",
    "confidence": 0.82,
    "bbox": [120, 80, 350, 400],
    "frame": 1840
  }
]
```

`output/annotated.mp4`: original video with bounding boxes + category +
confidence drawn on every frame, plus a `!!! CATEGORY ALERT !!!` banner
while a category is temporally confirmed.

## 11. Known limitations

- **This is zero-shot.** YOLO-World was not trained specifically on
  your CCTV footage or on these exact 5 categories -- expect it to
  miss things and occasionally misfire, especially on FALLEN_PERSON
  and FALLEN_TREE, which are events/poses rather than simple objects.
- Headlights, red/orange lighting, and reflections can still be
  mistaken for fire in a single frame -- temporal filtering reduces
  but does not eliminate this.
- Steam and fog can resemble smoke.
- No object tracking (no ByteTrack) -- each sampled frame is
  detected independently; the temporal layer is presence-based, not
  identity-based.
- No pose estimation -- FALLEN_PERSON relies purely on the prompt
  "person lying on the ground" plus persistence, not skeletal geometry.
- Weapon detection does not attempt person-weapon association in V1;
  it just flags visible weapons.
- Runs well below real-time on CPU; that's expected and acceptable for
  this prototype.

## 12. Adding training later

Once you've run this on real footage and logged failure cases (missed
fires, false SMOKE alerts, etc.), the natural next step is:

1. Save the frames/clips where V1 got it wrong.
2. Label just those hard cases.
3. Fine-tune YOLO-World (or a YOLOv8 detector) on that small, targeted
   dataset instead of starting from a large generic one.

None of that is part of this V1 -- this version exists purely to
establish whether pretrained + prompted YOLO-World is a viable
starting point before investing in labeling and training.
