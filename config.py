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
    # open flame (existing coverage)
    "fire",
    "flames",
    "burning fire",
    "house on fire",
    "fire engulfing a building",

    # explosion / blast
    "explosion",
    "fireball",
    "blast explosion with fire",
    "smoke and fire from an explosion",

    # electrical / short-circuit
    "electrical fire",
    "sparking wire",
    "electrical spark",
    "arcing electrical panel",
    "smoke coming from an electrical outlet",
    "burning electrical switchboard",
    "short circuit spark",
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
    # "dog attacking a person" REMOVED -- confirmed false positive via
    # batch_test.py: scored 0.912 on real_fight.mp4 (a human-only fight,
    # no dog present at all), HIGHER than the 0.441 true positive on the
    # actual dog_attack.mp4 clip. Full overlap, no threshold could fix
    # this -- the prompt itself is unreliable and needs to go.
    #
    # "person attacked by an animal" is the same prompt family and
    # hasn't been directly implicated yet, but treat it with the same
    # suspicion until tested -- if it starts false-firing on
    # human-only scenes, remove it too.
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
# MODEL-BASED CATEGORIES (multi_model_detector.py, not YOLO-World)
# ------------------------------------------------------------------
# These categories have no zero-shot prompts -- they come from small
# dedicated pretrained models loaded in multi_model_detector.py instead.
# WEAPON already exists above via YOLO-World prompts (covers knives);
# the pretrained firearm model contributes MORE detections into that
# SAME category rather than creating a new one -- guns only, knives
# still rely on the zero-shot prompts.
#
# main.py needs this list alongside CATEGORY_PROMPTS' keys to build the
# full report/temporal category set, since these categories aren't in
# CATEGORY_PROMPTS. temporal.py ALSO needs this list -- it was missed
# there originally, which meant FIGHT/ELECTRICAL could never actually
# get confirmed no matter how confidently the model detected them.
MODEL_BASED_CATEGORIES = ["FIGHT", "ELECTRICAL", "ACCIDENT"]

# Categories that are raw ingredients (person/dog/bag boxes), not real
# alert-ready events -- THEFT and DOG_ATTACK need a tracking/rule layer
# that doesn't exist yet (see multi_model_detector.py docstrings).
# main.py strips these out before they ever reach temporal.py, so a
# person walking near a dog doesn't become a confirmed "alert".
SIGNAL_ONLY_CATEGORIES = {"THEFT_SIGNAL", "DOG_ATTACK_SIGNAL"}

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
    # Added after batch_test.py showed FALLEN_TREE and WEAPON both
    # false-firing on car-crash/wreckage scenes -- "tree fallen on a
    # car" (0.488) and "tree blocking a road" (0.471) on
    # car_fire_accident.mp4/car_accident.mp4, neither of which has a
    # real tree. Dark, irregular wreckage/debris shapes are being
    # confused with fallen-tree and weapon silhouettes. No fallen-tree
    # true-positive clip exists yet to properly calibrate a threshold
    # fix, so suppression is the safer interim fix.
    "VEHICLE_WRECKAGE": [
        "car wreckage",
        "crashed car",
        "burnt car wreck",
        "car accident debris",
        "vehicle wreckage on road",
    ],
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

# ------------------------------------------------------------------
# CROSS-CATEGORY SUPPRESSION
# ------------------------------------------------------------------
# alert category -> list of {"category": suppressor, "iou": threshold}.
#
# Two DIFFERENT kinds of suppressor live in this same dict, and are
# consumed by two DIFFERENT places in the pipeline:
#
#   1. Suppressor-only categories (in SUPPRESSOR_CATEGORIES, e.g. ANIMAL)
#      -- detected by YOLO-World alongside everything else, never an
#      alert on their own. Handled entirely inside detector.py's
#      _suppress_cross_category(), which only ever sees zero-shot
#      detections. That's fine for ANIMAL since it has no multi-model
#      equivalent.
#
#   2. Real alert categories used as suppressors (FIRE, SMOKE) -- e.g. a
#      car-fire clip's smoke plume was scoring 0.92+ as FALLEN_TREE and
#      its flame/debris shapes were scoring 0.83+ as WEAPON. FIRE/SMOKE
#      can come from EITHER the zero-shot prompts OR the Thalos model
#      (multi_model_detector.py), and those two sources only get merged
#      together in main.py, AFTER detector.py's infer_raw() has already
#      returned. So this suppression case is handled in main.py, using
#      main.py's own _suppress_by_confirmed_events(), after the merge --
#      that way a fire caught ONLY by Thalos (and missed by zero-shot,
#      or vice versa) still correctly suppresses WEAPON/FALLEN_TREE.
#      A FIRE/SMOKE detection only counts as a suppressor once it clears
#      ITS OWN real alert threshold (via main.py's existing
#      _threshold_for() logic) -- floor-level noise shouldn't be able to
#      suppress a detection in another category.
#
# Was previously a single (category, iou) tuple per target -- widened to
# a list so a target (like WEAPON) can be suppressed by more than one
# real-world cause.
SUPPRESSED_BY = {
    "FALLEN_PERSON": [
        {"category": "ANIMAL", "iou": 0.3},
    ],
    "FALLEN_TREE": [
        {"category": "FIRE", "iou": 0.2},
        {"category": "SMOKE", "iou": 0.2},
        {"category": "VEHICLE_WRECKAGE", "iou": 0.2},
    ],
    "WEAPON": [
        {"category": "FIRE", "iou": 0.2},
        {"category": "SMOKE", "iou": 0.2},
        {"category": "VEHICLE_WRECKAGE", "iou": 0.2},
    ],
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
    "SMOKE": 0.35,    "WEAPON": 0.25,
    "FALLEN_PERSON": 0.30,
    "FALLEN_TREE": 0.30,
    # FIGHT recalibrated via batch_test.py against real footage:
    #   true positive (real_fight.mp4)              = 0.683
    #   false positives (police_none/dog_attack/
    #   car_fire_accident.mp4, all non-fight scenes) = 0.429-0.511
    # Raised from the model's own default (0.40, which sat INSIDE the
    # false-positive range) to 0.60 -- splits the gap with margin below
    # the one true positive observed. Only one true-positive data point
    # so far -- same "first pass, watch for more data" caveat FIRE/SMOKE
    # had before their own multi-round recalibration.
    "FIGHT": 0.60,
    # ELECTRICAL: no active detector exists yet (see multi_model_detector.py) --
    # this threshold is currently unreachable/irrelevant until one is wired up.
    "ELECTRICAL": 0.30,
    # ACCIDENT: UNVALIDATED starting guess -- road-accident-qqade
    # (multi_model_detector.py) just got wired in this session, no real
    # confidence numbers yet. Run debug_accident.py against
    # car_accident.mp4 / car_fire_accident.mp4 (true positive) and a
    # hard-negative clip, THEN replace this with a derived value the
    # same way FIRE/SMOKE/FIGHT were, instead of trusting this guess.
    "ACCIDENT": 0.40,
}

# ------------------------------------------------------------------
# PER-SOURCE CONFIDENCE OVERRIDES
# ------------------------------------------------------------------
# FIRE and SMOKE now get detections from TWO different models: the
# zero-shot YOLO-World prompts above, and multi_model_detector.py's
# Thalos model. These are not on the same confidence scale -- on real
# fire footage (debug_compare.py, 2026), Thalos's OWN true-positive
# range was fire=0.056-0.238, smoke=0.108-0.364. Zero-shot SMOKE's
# threshold (0.35) was calibrated against ITS OWN false positive
# (0.217, ceiling lights) -- sharing that number with Thalos would
# reject most of Thalos's real fire footage while barely giving it
# more margin than the zero-shot model's own noise floor.
#
# Matched against the START of Detection.prompt (main.py checks this
# before falling back to CONFIDENCE_THRESHOLDS[category]). These are
# FIRST-PASS values from one clip, not fully re-derived yet -- same
# "not yet trusted" caveat as FIGHT/ELECTRICAL above. Re-test against
# more footage, including hard negatives, before trusting these fully.
#
# WEAPON is in the SAME unresolved situation FIRE/SMOKE were in before
# this fix -- it has two sources (zero-shot prompts AND the dedicated
# firearm_yolo model) sharing one CONFIDENCE_THRESHOLDS["WEAPON"] value
# (0.25), with no entry here yet. Run debug_weapon.py against gun.mp4
# and police_none.mp4 (a true-positive and a hard-negative clip) to get
# real firearm_yolo confidence numbers, THEN add a
# "firearm_yolo:gun" entry here the same way "thalos:fire" was added --
# don't guess a number without that data, per the project's own rule
# (section 13).
SOURCE_CONFIDENCE_THRESHOLDS = {
    # thalos:fire/thalos:smoke are DEPRECATED -- see multi_model_detector.py.
    # debug_candidate.py showed thalos:fire's true positives (0.056-0.320)
    # fully overlap its false positives (0.063-0.327) -- no threshold
    # fixes that. Kept here only so old alerts.json data from before the
    # swap still makes sense if you look back at it.
    "thalos:fire": 0.05,
    "thalos:smoke": 0.10,

    # yolov10fire = TommyNgx/YOLOv10-Fire-and-Smoke-Detection, replacing
    # thalos as of this session. debug_candidate.py results:
    #   FIRE  true positives up to 0.804 (n=587) -- real false positives
    #         (excluding near-floor noise and the still-unconfirmed
    #         electric_shock.mp4 SMOKE reading) topped out at 0.472
    #         (police_none.mp4, n=3)
    #   SMOKE true positives up to 0.698 (n=245) -- real false positives
    #         topped out at 0.173 (real_fight.mp4, n=2)
    # Thresholds set with margin above the real false-positive ceiling.
    # FIRST PASS -- small false-positive sample sizes (n=3, n=2), same
    # "watch for more data" caveat as FIGHT's threshold. Re-derive once
    # more footage is available, and confirm whether electric_shock.mp4's
    # SMOKE reading (up to 0.577, n=60) is a real false positive or a
    # correct detection your ground-truth labels just didn't expect.
    "yolov10fire:Fire": 0.50,
    "yolov10fire:Smoke": 0.30,
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
# NOTE: this filter is shape-only and does NOT distinguish "tree lying
# down" from "wide dark plume of smoke" -- both produce a wide/short
# box. That's exactly why FALLEN_TREE also needed the new FIRE/SMOKE
# cross-category suppression above; aspect ratio alone let a car-fire
# smoke plume through as a 0.92+ FALLEN_TREE false positive.
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
    # FIGHT/ELECTRICAL detections come from multi_model_detector.py, not
    # detector.py's infer_raw() -- this dict is only ever read inside
    # detector.py, so these two entries are documentation, not active
    # filtering. If FIGHT ever needs a geometric filter (e.g. two
    # overlapping person-boxes rather than aspect ratio), it'd need to
    # be applied inside FightDetector.detect() itself.
    "FIGHT": None,
    "ELECTRICAL": None,
    # ACCIDENT also comes from multi_model_detector.py (AccidentDetector),
    # not detector.py's infer_raw() -- documentation only, same as
    # FIGHT/ELECTRICAL above.
    "ACCIDENT": None,
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
    # New model-based categories -- unvalidated starting points, same
    # caveat as CONFIDENCE_THRESHOLDS above.
    "FIGHT": 2,
    "ELECTRICAL": 2,
    # ACCIDENT: same unvalidated-starting-point caveat as above.
    "ACCIDENT": 2,
}
# ------------------------------------------------------------------
# ALERT COOLDOWN
# ------------------------------------------------------------------
# Once a category is confirmed, don't fire another alert for the same
# category for this many seconds (status overlay still shows though).
ALERT_COOLDOWN_SECONDS = 20