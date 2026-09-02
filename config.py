"""
config.py
All tunable settings live here. Edit this file to change model, prompts,
thresholds, temporal logic, or paths -- no other file needs to change.
"""

# ------------------------------------------------------------------
# MODEL
# ------------------------------------------------------------------
# Smallest practical YOLO-World checkpoint for CPU use.
# Ultralytics will auto-download this on first run if not present locally.
MODEL_PATH = "yolov8s-worldv2.pt"

# No CUDA on this machine -- always CPU.
DEVICE = "cpu"

# ------------------------------------------------------------------
# VIDEO PROCESSING
# ------------------------------------------------------------------
# How many frames per second to actually run inference on.
# The rest of the frames are copied through to the output video unmodified
# (with the last known detection overlay held over them).
PROCESS_FPS = 2

INPUT_PATH = "input/test.mp4"
OUTPUT_PATH = "output/annotated.mp4"
ALERTS_PATH = "output/alerts.json"
REPORT_PATH = "output/report.html"

# How many of the best raw detections (by confidence) to keep as
# thumbnail images per category in the HTML report.
REPORT_TOP_K_PER_CATEGORY = 3

# ------------------------------------------------------------------
# PROMPTS
# ------------------------------------------------------------------
# Each low-level prompt maps back to one of the 5 high-level categories.
# Add/remove/edit prompts freely -- detector.py reads this automatically.

# NOTE ON PROMPT PHRASING: YOLO-World's confidence is sensitive to exact
# wording -- short single/two-word prompts ("fire", "gun") and longer
# descriptive phrases ("person holding a rifle") can score very
# differently on the same object. Keep both styles per category so the
# vocabulary has multiple ways to "catch" the same real-world event.

FIRE_PROMPTS = [
    "fire",
    "flames",
    "open flame",
    "burning fire",
    "house on fire",
    "building on fire",
    "large fire",
    "orange flames",
    "wildfire",
    "blaze",
    "burning building",
    "fire breaking out",
    "small fire starting",
    "flames and smoke",
    # ignition / electrical-fault precursors to a fire -- these don't
    # look like "flames" but are the earliest visual sign of one
    "electrical spark",
    "sparks from wires",
    "sparking electrical panel",
    "sparking switchboard",
    "electrical fire starting",
]

SMOKE_PROMPTS = [
    "smoke",
    "smoke cloud",
    "thick smoke",
    "black smoke",
    "white smoke",
    "grey smoke",
    "smoke from a building",
    "smoke rising",
    "thin smoke",
    "wisp of smoke",
    # "haze in the air" removed -- on test footage it was the likely
    # culprit for flagging bright ceiling light fixtures as smoke
    # (0.217 confidence in a hallway with no smoke at all)
    #
    # smaller-scale / indoor smoke sources -- distinct visually from a
    # building fire's thick black smoke plume. These correctly caught
    # a switchboard spark+smoke case (0.929 confidence) that earlier
    # prompt sets missed entirely -- keep them.
    "smoke from an electrical panel",
    "smoke from a switchboard",
    "smoke from an outlet",
    "smoke from equipment",
    "smoky room",
]

WEAPON_PROMPTS = [
    "gun",
    "handgun",
    "pistol",
    "rifle",
    "assault rifle",
    "knife",
    "weapon",
    "firearm",
    # descriptive phrases score differently than the bare noun, and are
    # often a better match when the weapon is partially held/obscured
    "person holding a gun",
    "person holding a rifle",
    "person carrying a rifle",
    "man with a rifle",
    "person aiming a weapon",
    "concealed weapon",
]

FALLEN_PERSON_PROMPTS = [
    "person lying on the ground",
    "fallen person",
    "person on the ground",
    "person collapsed",
    "person knocked down",
    "person falling down",
    # a fall from an attack/struggle looks visually different (motion
    # blur, twisted pose) from someone calmly lying down -- these
    # phrases target that case specifically
    "person struggling on the ground",
    "person being attacked",
    "dog attacking a person",
    "person attacked by an animal",
    "injured person on the ground",
    "person lying motionless",
]

FALLEN_TREE_PROMPTS = [
    "fallen tree",
    "tree lying on the ground",
    "tree blocking a road",
    "uprooted tree",
    "tree collapsed",
    "broken tree branch on the ground",
    "tree fallen on a car",
    "storm damaged tree",
]

# category -> list of prompts (used to build the YOLO-World vocabulary)
CATEGORY_PROMPTS = {
    "FIRE": FIRE_PROMPTS,
    "SMOKE": SMOKE_PROMPTS,
    "WEAPON": WEAPON_PROMPTS,
    "FALLEN_PERSON": FALLEN_PERSON_PROMPTS,
    "FALLEN_TREE": FALLEN_TREE_PROMPTS,
}

# ------------------------------------------------------------------
# SUPPRESSOR PROMPTS (not alert categories -- cross-checks only)
# ------------------------------------------------------------------
# Some real-world objects are geometrically indistinguishable from an
# alert category at the bounding-box level -- e.g. a standing dog's box
# is wide-and-short, the SAME shape as a fallen person's box, so
# MIN_ASPECT_RATIO can't tell them apart (on test footage a standing
# dog scored 0.989 as FALLEN_PERSON). These prompts are added to the
# model's vocabulary and detected like any other, but they never become
# an alert on their own -- detector.py uses them only to reject an
# alert-category detection that overlaps one of them in the same frame.
SUPPRESSOR_PROMPTS = {
    "ANIMAL": ["dog", "cat", "stray dog", "animal"],
}

# Every prompt sent to the model = alert-category prompts + suppressor
# prompts. Both need to be in the vocabulary for suppression to work.
_ALL_CATEGORY_PROMPTS = {**CATEGORY_PROMPTS, **SUPPRESSOR_PROMPTS}

# reverse lookup: exact prompt string -> category (built automatically)
PROMPT_TO_CATEGORY = {
    prompt: category
    for category, prompts in _ALL_CATEGORY_PROMPTS.items()
    for prompt in prompts
}

# category names that exist only to suppress other categories' false
# positives -- detector.py strips these out before returning results,
# so nothing downstream (temporal tracker, alerts, report) ever sees them
SUPPRESSOR_CATEGORIES = set(SUPPRESSOR_PROMPTS.keys())

# alert category -> (suppressor category, IoU threshold). If a
# detection in the alert category overlaps a same-frame suppressor
# detection by at least this much, the alert-category detection is
# dropped as a likely false positive.
SUPPRESSED_BY = {
    "FALLEN_PERSON": ("ANIMAL", 0.3),
}

# Flat list passed to model.set_classes()
ALL_PROMPTS = list(PROMPT_TO_CATEGORY.keys())

# ------------------------------------------------------------------
# CONFIDENCE THRESHOLDS (per category, recalibrated from real test footage)
# ------------------------------------------------------------------
# Round 2 recalibration. We now have BOTH true-positive and false-positive
# confidence values for several categories from real test footage, so
# thresholds below are set to sit between them instead of guessing:
#
#   SMOKE:  true positive (switchboard spark+smoke) = 0.929
#           false positive (hallway ceiling lights)  = 0.217
#           -> threshold set well above the false positive, well below the true one
#   WEAPON: true positive (carried rifle)            = 0.391
#           false positive (fire debris/embers)      = 0.218 (didn't confirm, but close)
#           -> threshold raised to build in a safety margin, not just luck
#   FIRE:   still capped at ~0.07 on real fire footage across two separate
#           tests -- lowered further since no competing false-positive
#           signal has been observed yet at this level. If false fire
#           alerts start appearing, raise this back up.
#   FALLEN_PERSON / FALLEN_TREE: confidence alone can't separate real
#           events from false ones here -- a normal standing person
#           scored 0.806, HIGHER than a genuine fall (0.428), and
#           upright standing trees scored 0.402. Thresholds are left
#           roughly as before; the new MIN_ASPECT_RATIO geometric filter
#           below is now the primary defense for these two categories.
CONFIDENCE_THRESHOLDS = {
    "FIRE": 0.065,
    "SMOKE": 0.35,
    "WEAPON": 0.25,
    "FALLEN_PERSON": 0.30,
    "FALLEN_TREE": 0.30,
}

# ------------------------------------------------------------------
# GEOMETRIC SANITY FILTERS (per category, optional)
# ------------------------------------------------------------------
# FALLEN_PERSON and FALLEN_TREE describe an ORIENTATION/STATE (fallen ==
# lying down / horizontal), not just an object's presence -- and
# confidence scores don't reliably encode that. On real test footage,
# "person on the ground" scored 0.806 on an ordinary standing/walking
# person, higher than the 0.428 it scored on someone who had genuinely
# fallen; "fallen tree" scored 0.402 on upright standing trees. No
# threshold can separate those since the false positive outscores the
# true one.
#
# A cheap, reliable extra signal is the detected bounding box's aspect
# ratio: a standing person's or a standing tree's box is tall and
# narrow (height >> width, i.e. width/height is small). A person lying
# down or a tree lying on the ground produces a wide, short box
# (width/height is at or above 1).
#
# MIN_ASPECT_RATIO[category] = minimum width/height ratio a detection's
# bbox must have to be kept for that category. None = no geometric
# filter (used where orientation isn't meaningful: FIRE/SMOKE/WEAPON).
# Tune these against your own footage -- they're a starting point.
MIN_ASPECT_RATIO = {
    "FIRE": None,
    "SMOKE": None,
    "WEAPON": None,
    "FALLEN_PERSON": 0.75,  # reject tall/narrow, standing-person-shaped boxes
    "FALLEN_TREE": 1.0,     # reject tall/narrow, standing-trunk-shaped boxes
}

# The floor passed straight to YOLO-World's own predict() call.
# This MUST be lower than every value in CONFIDENCE_THRESHOLDS above,
# otherwise Ultralytics silently discards low-scoring boxes before your
# per-category thresholds ever get a chance to see them. Category
# filtering happens in detector.py/main.py, not here -- this only
# controls what the model is allowed to report at all.
RAW_CONFIDENCE_FLOOR = 0.05

# ------------------------------------------------------------------
# TEMPORAL CONFIRMATION
# ------------------------------------------------------------------
# Sliding window (in sampled/processed frames, not raw video frames) and
# how many "hits" inside that window are needed before an alert fires.
# Widened from 5 -> 7 sampled frames (at PROCESS_FPS=2 that's 3.5s
# instead of 2.5s) to give fast/jittery events (a struggle, a fall, an
# intermittently-visible carried weapon) more room to accumulate hits
# without needing every single sampled frame to clear threshold.
TEMPORAL_WINDOW = 7

MIN_HITS = {
    "FIRE": 2,
    "SMOKE": 3,
    "WEAPON": 2,
    # lowered from 3 -> 2: a dog attack / struggle is brief and the
    # person's pose changes rapidly, so confidence bounces around --
    # requiring 3 consistent hits was filtering out real events
    "FALLEN_PERSON": 2,
    "FALLEN_TREE": 3,
}

# ------------------------------------------------------------------
# ALERT COOLDOWN
# ------------------------------------------------------------------
# Once a category is confirmed, don't fire another alert for the same
# category for this many seconds (status overlay still shows though).
ALERT_COOLDOWN_SECONDS = 20