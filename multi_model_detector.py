"""
One small model per category instead of one YOLO-World model doing
everything. Each wrapper exposes the same .detect(frame) -> list[Detection]
interface so temporal.py/alerts.py don't need to know which backend
produced a detection.

Add/remove categories by adding/removing entries in build_registry() --
nothing else in your pipeline needs to change.
"""
from pathlib import Path
import requests

import config

# Reuse detector.py's own Detection class rather than defining a new
# shape -- this is the exact object temporal.py/alerts.py/video.py/
# report.py already know how to handle (category, confidence, bbox,
# prompt). "prompt" here holds the source model + its own class name
# (e.g. "thalos:open_flame") instead of a YOLO-World text prompt --
# same field, different origin, useful for debugging which model fired.
from detector import Detection


# ---------------------------------------------------------------------
# FIRE / SMOKE / SPARK -- Thalos Fire Safety v1 (YOLOv8-derived, HF hosted)
# Classes: open_flame, smoke, fire_risk, spark, ignition_source
# License: AGPL-3.0 -- check against your client-deliverable terms before shipping.
#
# *** KNOWN ISSUE (batch_test.py, this session) ***
# thalos:fire's confidence does NOT reliably separate true from false
# positives -- true positives scored 0.056-0.320, false positives
# scored 0.063-0.327 on unrelated clips (gun.mp4, police_none.mp4,
# real_fight.mp4). The real house fire (fire_house.mp4) scored LOWER
# (0.056) than two of the false positives. No threshold can fix this --
# the model's own confidence output isn't discriminative on this
# footage style. STILL LEFT ACTIVE for now because it's currently the
# ONLY thing producing correct FIRE/SMOKE alerts on fire_house.mp4 and
# forest_fire.mp4 (zero-shot's own contribution gets hidden by
# detector._dedupe() whenever thalos scores higher on the same box) --
# disabling it outright would likely cost real recall with no tested
# fallback. Priority: find and swap in a properly-evaluated fire/smoke
# model (one with an actual mAP/precision-recall table on its model
# card, not just download/like counts) rather than tuning this one
# further.
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# FIRE / SMOKE -- TommyNgx/YOLOv10-Fire-and-Smoke-Detection (YOLOv8-based,
# HF hosted, GATED repo -- requires accepting terms + an HF token).
# Classes: Fire, Smoke (note capitalization -- verified via
# debug_candidate.py, don't assume lowercase like the old thalos model).
#
# REPLACED FireSafetyDetector (Thalos) this session. debug_candidate.py
# showed Thalos's true/false positive confidence ranges fully
# overlapped (unsalvageable by any threshold) -- this model showed real
# separation instead: true positives up to 0.804 (FIRE) / 0.698 (SMOKE),
# with real false positives well below that (see config.py's
# SOURCE_CONFIDENCE_THRESHOLDS comment for the exact numbers and the
# one still-unconfirmed reading on electric_shock.mp4).
#
# Old FireSafetyDetector class kept further down, not deleted, in case
# this one needs to be reverted or compared again later.
# ---------------------------------------------------------------------
class FireSmokeDetectorV2:
    HF_REPO = "TommyNgx/YOLOv10-Fire-and-Smoke-Detection"
    HF_FILENAME = "best.pt"
    WEIGHTS_PATH = Path("weights/yolov10_fire_smoke.pt")

    CLASS_TO_CATEGORY = {
        "fire": "FIRE",
        "smoke": "SMOKE",
    }

    def __init__(self, conf_threshold: float = config.RAW_CONFIDENCE_FLOOR, hf_token: str | None = None):
        from ultralytics import YOLO
        from huggingface_hub import hf_hub_download
        import os
        if not self.WEIGHTS_PATH.exists():
            self.WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            token = hf_token or os.environ.get("HF_TOKEN")
            downloaded = hf_hub_download(repo_id=self.HF_REPO, filename=self.HF_FILENAME, token=token)
            Path(downloaded).rename(self.WEIGHTS_PATH)
        self.model = YOLO(str(self.WEIGHTS_PATH))
        self.conf_threshold = conf_threshold

    def detect(self, frame) -> list[Detection]:
        results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
        out = []
        for box in results.boxes:
            cls_name = results.names[int(box.cls)]  # "Fire" / "Smoke", capitalized
            category = self.CLASS_TO_CATEGORY.get(cls_name.lower())
            if category is None:
                continue
            out.append(Detection(
                category=category,
                confidence=float(box.conf),
                bbox=tuple(box.xyxy[0].tolist()),
                # Prompt prefix MUST match config.SOURCE_CONFIDENCE_THRESHOLDS
                # keys exactly, e.g. "yolov10fire:Fire" -- keep the model's
                # real capitalization here so debugging output matches
                # what debug_candidate.py already showed you.
                prompt=f"yolov10fire:{cls_name}",
            ))
        return out


# ---------------------------------------------------------------------
# OLD: FIRE / SMOKE / SPARK -- Thalos Fire Safety v1. DISABLED, not
# deleted -- see FireSmokeDetectorV2 above, which replaced this.
# ---------------------------------------------------------------------
class FireSafetyDetector:
    HF_URL = "https://huggingface.co/thalostech2025/thalos-fire-safety-v1/resolve/main/fire_weights.pt"
    WEIGHTS_PATH = Path("weights/fire_weights.pt")

    # Map the model's own class names to your alert categories.
    CLASS_TO_CATEGORY = {
        "fire": "FIRE",
        "smoke": "SMOKE",
    }

    def __init__(self, conf_threshold: float = config.RAW_CONFIDENCE_FLOOR):
        from ultralytics import YOLO
        if not self.WEIGHTS_PATH.exists():
            self.WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            resp = requests.get(self.HF_URL)
            resp.raise_for_status()
            self.WEIGHTS_PATH.write_bytes(resp.content)
        self.model = YOLO(str(self.WEIGHTS_PATH))
        self.conf_threshold = conf_threshold

    def detect(self, frame) -> list[Detection]:
        results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
        out = []
        for box in results.boxes:
            cls_name = results.names[int(box.cls)]
            category = self.CLASS_TO_CATEGORY.get(cls_name)
            if category is None:
                continue
            out.append(Detection(
                category=category,
                confidence=float(box.conf),
                bbox=tuple(box.xyxy[0].tolist()),
                prompt=f"thalos:{cls_name}",
            ))
        return out


# ---------------------------------------------------------------------
# ELECTRICAL (arc/spark on panels/wiring) -- Roboflow-hosted inference.
# https://universe.roboflow.com/a-a-fbw8p/fire-smoke-and-spark-detection
# model_id = "fire-smoke-and-spark-detection/2"
# mAP@50 89.3%, precision 86.4%, recall 86.7% (THEIR benchmark, not
# your footage -- unvalidated until debug_electrical.py is run).
#
# This model's 8 classes are messy/duplicated: "Fire","fire","Smoke",
# "smoke","6","Fire-Smoke","spark","sparks". Only "spark"/"sparks" are
# used here for ELECTRICAL -- the fire/smoke classes are deliberately
# IGNORED, since FIRE/SMOKE already has a properly-calibrated dedicated
# model (FireSmokeDetectorV2) and mixing in a second, uncalibrated
# fire/smoke source would just add noise. "6" is an unexplained/likely
# mislabeled class -- also ignored.
# ---------------------------------------------------------------------
class ElectricalArcDetector:
    def __init__(self, api_key: str, workspace: str = "a-a-fbw8p",
                 project: str = "fire-smoke-and-spark-detection", version: int = 2,
                 conf_threshold: float = config.RAW_CONFIDENCE_FLOOR):
        from inference_sdk import InferenceHTTPClient, InferenceConfiguration
        self.client = InferenceHTTPClient(api_url="https://serverless.roboflow.com", api_key=api_key)
        # *** FIX (same bug as AccidentDetector originally had) *** --
        # confidence MUST be sent to the server via .configure(), not
        # just checked locally after the fact, or real detections below
        # Roboflow's own server-side default threshold never come back
        # at all.
        self.client.configure(InferenceConfiguration(
            confidence_threshold=conf_threshold,
            api_key_transport="header",
        ))
        self.model_id = f"{workspace}/{project}/{version}"
        self.conf_threshold = conf_threshold

    def detect(self, frame) -> list[Detection]:
        # inference_sdk expects a file path or ndarray; adapt to however
        # main.py currently passes frames (cv2 ndarray shown here).
        result = self.client.infer(frame, model_id=self.model_id)
        out = []
        for pred in result.get("predictions", []):
            if pred["confidence"] < self.conf_threshold:
                continue
            if pred["class"].lower() not in ("spark", "sparks", "arc"):
                continue
            cx, cy, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
            bbox = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            out.append(Detection(category="ELECTRICAL", confidence=pred["confidence"], bbox=bbox,
                                  prompt=f"roboflow_arc:{pred['class']}"))
        return out


# ---------------------------------------------------------------------
# ACCIDENT (road/vehicle accidents) -- Roboflow-hosted inference.
# https://universe.roboflow.com/nathanresearch-tnuhn/road-accident-qqade
# model_id = "road-accident-qqade/1", single class: "Accident"
# yolov5-based, CC BY 4.0. Project page states mAP@50 75.4% / precision
# 68.6% / recall 74.8% -- that's THEIR benchmark, not your footage.
# UNVALIDATED on your test set until debug_accident.py is run against a
# true-positive clip (car_accident.mp4 / car_fire_accident.mp4) and a
# hard-negative clip -- same rule as every other model here (section 13):
# don't trust a threshold until you've derived it from real confidence
# numbers.
#
# NOTE: car_fire_accident.mp4 will likely also fire FIRE/SMOKE from
# FireSmokeDetectorV2 on the same clip -- that's expected, not a bug.
# ACCIDENT is NOT currently in config.SUPPRESSED_BY as a suppressor
# target, so it won't be suppressed by FIRE/SMOKE the way WEAPON/
# FALLEN_TREE are. Watch debug_accident.py's output on that clip --
# if wreckage/smoke shapes cause ACCIDENT false positives the way they
# did for WEAPON/FALLEN_TREE, it may need the same treatment (either a
# SOURCE_CONFIDENCE_THRESHOLDS entry with a high margin, or adding
# ACCIDENT to SUPPRESSED_BY under FIRE/SMOKE).
#
# *** FIX (this session) *** -- .infer() was being called with NO
# confidence_threshold sent to Roboflow's server at all. self.conf_threshold
# was only applied AFTER the response came back, but Roboflow's hosted
# inference has its OWN server-side default confidence threshold -- any
# real detection below that default never gets returned in the first
# place, so the local floor never got a chance to matter. This is why
# debug_accident.py returned zero detections on every sampled frame of
# car_accident.mp4, a clip that supposedly has a real accident in it --
# not necessarily a "the model doesn't work" result, likely a "the
# server silently threw away everything before we ever saw it" result.
# Fixed by explicitly configuring the client's confidence_threshold
# before calling .infer(), same fix documented in Roboflow's own forum
# for this exact symptom. ElectricalArcDetector above has this SAME
# latent bug (never configured, never tested) -- not fixed here since
# it's unrelated to today's task, but flag it before trusting that one.
# ---------------------------------------------------------------------
class AccidentDetector:
    def __init__(self, api_key: str, conf_threshold: float = config.RAW_CONFIDENCE_FLOOR):
        from inference_sdk import InferenceHTTPClient, InferenceConfiguration
        self.client = InferenceHTTPClient(api_url="https://serverless.roboflow.com", api_key=api_key)
        self.client.configure(InferenceConfiguration(
            confidence_threshold=conf_threshold,
            api_key_transport="header",  # silences the legacy-auth deprecation warning
        ))
        self.model_id = "road-accident-qqade/1"
        self.conf_threshold = conf_threshold

    def detect(self, frame) -> list[Detection]:
        result = self.client.infer(frame, model_id=self.model_id)
        out = []
        for pred in result.get("predictions", []):
            if pred["confidence"] < self.conf_threshold:
                continue
            if pred["class"] != "Accident":  # only class this model has, but stay explicit
                continue
            cx, cy, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
            bbox = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            out.append(Detection(
                category="ACCIDENT",
                confidence=pred["confidence"],
                bbox=bbox,
                # prefix MUST match a future config.SOURCE_CONFIDENCE_THRESHOLDS
                # key exactly, same convention as roboflow_arc/yolov10fire above
                prompt=f"roboflow_accident:{pred['class']}",
            ))
        return out


# ---------------------------------------------------------------------
# FIGHT / VIOLENCE -- pretrained YOLOv8-nano, MIT license.
# Classes: Violence/Fight, NoViolence/NoFight (only Fight is emitted here).
# https://huggingface.co/Musawer14/fight_detection_yolov8
#
# batch_test.py result (this session): true positive 0.683
# (real_fight.mp4), false positives 0.429-0.511 (police_none.mp4,
# dog_attack.mp4, car_fire_accident.mp4). Clean gap between the two --
# config.CONFIDENCE_THRESHOLDS["FIGHT"] raised from 0.40 to 0.60 to
# separate them. Kept ACTIVE (unlike WeaponDetector below) because,
# unlike that model, this one DID produce a correct true positive and
# has a workable threshold fix -- no model swap needed here, at least
# not yet. Still only one true-positive data point -- watch for more.
# ---------------------------------------------------------------------
class FightDetector:
    WEIGHTS_PATH = Path("weights/fight_yolo_nano.pt")

    def __init__(self, conf_threshold: float = config.RAW_CONFIDENCE_FLOOR):
        from ultralytics import YOLO
        from huggingface_hub import hf_hub_download
        if not self.WEIGHTS_PATH.exists():
            self.WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            downloaded = hf_hub_download(
                repo_id="Musawer14/fight_detection_yolov8",
                filename="Yolo_nano_weights.pt",
            )
            Path(downloaded).rename(self.WEIGHTS_PATH)
        self.model = YOLO(str(self.WEIGHTS_PATH))
        self.conf_threshold = conf_threshold

    def detect(self, frame) -> list[Detection]:
        results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
        out = []
        for box in results.boxes:
            cls_name = results.names[int(box.cls)]
            if "violence" not in cls_name.lower() and "fight" not in cls_name.lower():
                continue
            if "no" in cls_name.lower():  # skip NoViolence/NoFight class
                continue
            out.append(Detection(category="FIGHT", confidence=float(box.conf), bbox=tuple(box.xyxy[0].tolist()),
                                  prompt=f"fight_yolo:{cls_name}"))
        return out


# ---------------------------------------------------------------------
# FIGHT candidate #2 -- Roboflow-hosted, YOLOv11 KEYPOINT (pose) model,
# not a plain object detector like everything else here.
# https://universe.roboflow.com/neuron-x1sgr/fight-detection-7xdy7
# model_id = "fight-detection-7xdy7/2" -- NOTE: the URL you sent pointed
# at /dataset/5, which is only a labeled dataset VERSION with no trained
# weights attached (Models: 0 on that version). The actual deployed,
# trained model lives at version 2 instead -- used here.
# mAP@50 83.7%, precision 78.9%, recall 81.6% (THEIR benchmark).
#
# Single class "Fight-Detection". Being a pose model, predictions may
# include a "keypoints" list alongside the usual x/y/width/height/
# confidence/class fields -- keypoints are ignored here, only the
# bounding box + confidence are used, same shape as every other
# Detection in this file.
#
# UNTESTED. The currently-ACTIVE FightDetector (above, local HF model)
# already has one confirmed true positive (0.683 on real_fight.mp4)
# with a working threshold (0.60) -- this is a candidate to compare
# against it, not an automatic replacement. Don't swap FightDetector
# out in build_registry() until debug_fight_roboflow.py shows this one
# is actually better on your footage, not just on paper.
# ---------------------------------------------------------------------
class FightDetectorV2:
    def __init__(self, api_key: str, conf_threshold: float = config.RAW_CONFIDENCE_FLOOR):
        from inference_sdk import InferenceHTTPClient, InferenceConfiguration
        self.client = InferenceHTTPClient(api_url="https://serverless.roboflow.com", api_key=api_key)
        self.client.configure(InferenceConfiguration(
            confidence_threshold=conf_threshold,
            api_key_transport="header",
        ))
        self.model_id = "fight-detection-7xdy7/2"
        self.conf_threshold = conf_threshold

    def detect(self, frame) -> list[Detection]:
        result = self.client.infer(frame, model_id=self.model_id)
        out = []
        for pred in result.get("predictions", []):
            if pred["confidence"] < self.conf_threshold:
                continue
            cx, cy, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
            bbox = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            out.append(Detection(
                category="FIGHT",
                confidence=pred["confidence"],
                bbox=bbox,
                prompt=f"roboflow_fight:{pred['class']}",
            ))
        return out


# ---------------------------------------------------------------------
# WEAPON -- pretrained YOLOv8-nano firearm detector, single "Gun" class,
# 89.0% mAP@0.5 per model card.
# https://huggingface.co/Subh775/Firearm_Detection_Yolov8n
#
# *** DISABLED -- confirmed non-functional on our footage, batch_test.py ***
# Zero true positives across the whole test set (including gun.mp4,
# the one clip that actually has a firearm -- firearm_yolo produced NO
# detection there at all). Four false positives on completely unrelated
# clips: fire_house.mp4 (0.283), car_fire_accident.mp4 (0.728),
# dog_attack.mp4 (0.842), car_accident.mp4 (0.569). The model's
# advertised 89% mAP clearly doesn't transfer to this footage style.
#
# Zero-shot's own "firearm" prompt correctly caught the real gun at
# 0.397 with NO false positives observed anywhere else in the test set --
# so WEAPON coverage is NOT lost by disabling this, only the
# (non-functional) extra contribution from this specific model is
# removed. Not instantiated in build_registry() below until a properly
# evaluated replacement is found and re-tested via debug_weapon.py.
# Left in the file (not deleted) in case a future re-test on different
# footage shows it performing better, or a fine-tuned version becomes
# available.
# ---------------------------------------------------------------------
class WeaponDetector:
    def __init__(self, conf_threshold: float = config.RAW_CONFIDENCE_FLOOR):
        from ultralytics import YOLO
        from huggingface_hub import hf_hub_download
        weights_path = hf_hub_download(
            repo_id="Subh775/Firearm_Detection_Yolov8n",
            filename="weights/best.pt",
        )
        self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame) -> list[Detection]:
        results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
        out = []
        for box in results.boxes:
            out.append(Detection(category="WEAPON", confidence=float(box.conf), bbox=tuple(box.xyxy[0].tolist()),
                                  prompt="firearm_yolo:gun"))
        return out


# ---------------------------------------------------------------------
# WEAPON candidate #3 -- Roboflow-hosted, single "Weapons" class.
# https://universe.roboflow.com/weapon-detection-nl5fv/weapon-detection-db7n2
# model_id = "weapon-detection-db7n2/2"
# mAP@50 67.3%, precision 76.6%, recall 61.0% (THEIR benchmark).
#
# UNTESTED on your footage. The first two WEAPON candidates
# (WeaponDetector above, ThreatDetector below) both had HIGHER
# advertised benchmarks than this one (89.0% and 81.3% mAP respectively)
# and BOTH failed completely on real footage -- zero true positives,
# high-confidence false positives on unrelated clips. This candidate's
# lower benchmark number doesn't tell you anything about whether it'll
# do better or worse on YOUR footage -- that's exactly why
# debug_weapon_roboflow.py exists, run it before trusting this at all.
#
# Only class is "Weapons" (singular umbrella category covering
# guns/knives/etc together, unlike the local models which had separate
# classes) -- maps directly to your WEAPON category.
# ---------------------------------------------------------------------
class WeaponDetectorV2:
    def __init__(self, api_key: str, conf_threshold: float = config.RAW_CONFIDENCE_FLOOR):
        from inference_sdk import InferenceHTTPClient, InferenceConfiguration
        self.client = InferenceHTTPClient(api_url="https://serverless.roboflow.com", api_key=api_key)
        self.client.configure(InferenceConfiguration(
            confidence_threshold=conf_threshold,
            api_key_transport="header",
        ))
        self.model_id = "weapon-detection-db7n2/2"
        self.conf_threshold = conf_threshold

    def detect(self, frame) -> list[Detection]:
        result = self.client.infer(frame, model_id=self.model_id)
        out = []
        for pred in result.get("predictions", []):
            if pred["confidence"] < self.conf_threshold:
                continue
            cx, cy, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
            bbox = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            out.append(Detection(
                category="WEAPON",
                confidence=pred["confidence"],
                bbox=bbox,
                prompt=f"roboflow_weapon:{pred['class']}",
            ))
        return out


# ---------------------------------------------------------------------
# WEAPON candidate #2 -- Subh775/Threat-Detection-YOLOv8n. TESTED AND
# REJECTED via debug_candidate.py this session.
#
# Classes: Gun, explosion, grenade, knife. Advertised 81.3% mAP@50
# overall, 96.7% precision on Gun specifically. Real test results told
# a very different story: true positives (gun.mp4) ranged 0.179-0.715
# (n=8), but false positives across UNRELATED clips ranged 0.050-0.735
# (n=136) -- the false-positive ceiling (0.735, on fire_house.mp4) is
# actually HIGHER than the true-positive ceiling. Massive false-positive
# volume too: 17x more false detections than true ones across the test
# set. This is the SAME author as WeaponDetector above, which also
# failed -- two different models, same failure pattern (advertised
# benchmark doesn't transfer to real footage). Not instantiated in
# build_registry(). Kept here for reference only; don't re-enable
# without new evidence.
# ---------------------------------------------------------------------
class ThreatDetector:
    HF_REPO = "Subh775/Threat-Detection-YOLOv8n"
    HF_FILENAME = "weights/best.pt"
    WEIGHTS_PATH = Path("weights/threat_detection.pt")

    # Only Gun/grenade map to WEAPON -- explosion/knife are left
    # unmapped for now (knife would double up with zero-shot's existing
    # "knife" prompt coverage; explosion isn't a defined alert category).
    CLASS_TO_CATEGORY = {
        "gun": "WEAPON",
        "grenade": "WEAPON",
    }

    def __init__(self, conf_threshold: float = config.RAW_CONFIDENCE_FLOOR):
        from ultralytics import YOLO
        from huggingface_hub import hf_hub_download
        if not self.WEIGHTS_PATH.exists():
            self.WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            downloaded = hf_hub_download(repo_id=self.HF_REPO, filename=self.HF_FILENAME)
            Path(downloaded).rename(self.WEIGHTS_PATH)
        self.model = YOLO(str(self.WEIGHTS_PATH))
        self.conf_threshold = conf_threshold

    def detect(self, frame) -> list[Detection]:
        results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
        out = []
        for box in results.boxes:
            cls_name = results.names[int(box.cls)]
            category = self.CLASS_TO_CATEGORY.get(cls_name.lower())
            if category is None:
                continue
            out.append(Detection(
                category=category,
                confidence=float(box.conf),
                bbox=tuple(box.xyxy[0].tolist()),
                prompt=f"threat_yolo:{cls_name}",
            ))
        return out


# ---------------------------------------------------------------------
# THEFT -- there is no equivalent single-frame pretrained model.
# Reasoning: fire/smoke/spark are visual PATTERNS present in one frame.
# Theft is a BEHAVIOR across frames (person approaches item -> picks it
# up -> leaves without a transaction) -- a static detector can at best
# give you the raw ingredients (person, bag, register-area), not the
# event itself. Academic theft/shoplifting models exist (UCF-Crime-based
# I3D/C3D action recognition) but are heavy video-clip classifiers, not
# a drop-in YOLO .pt -- different deployment shape entirely, plan for
# that separately rather than expecting a small model here.
#
# Practical placeholder until you build that: COCO-pretrained YOLOv8
# already detects "person", "backpack", "handbag", "suitcase" out of the
# box (no fine-tuning needed) -- use it to feed a simple rule, not as a
# theft "detector" per se.
# ---------------------------------------------------------------------
class TheftHeuristicDetector:
    COCO_CLASSES = {"person", "backpack", "handbag", "suitcase"}

    def __init__(self, conf_threshold: float = 0.4):
        from ultralytics import YOLO
        self.model = YOLO("yolov8n.pt")  # plain COCO weights, no download needed beyond first pull
        self.conf_threshold = conf_threshold

    def detect(self, frame) -> list[Detection]:
        """Returns raw person/bag boxes tagged as category 'THEFT_SIGNAL' --
        NOT an alert-ready category. You still need a rule/tracking layer
        (e.g. ByteTrack: does a person+bag pair that entered together leave
        the frame separately, near a shelf/register zone?) before this
        becomes a real THEFT alert. Don't wire this straight into
        temporal.py the way FIRE/SMOKE/ELECTRICAL are -- it needs an
        identity-tracking step first, which is a bigger addition to
        detector.py than swapping a model."""
        results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
        out = []
        for box in results.boxes:
            cls_name = results.names[int(box.cls)]
            if cls_name not in self.COCO_CLASSES:
                continue
            out.append(Detection(
                category="THEFT_SIGNAL",
                confidence=float(box.conf),
                bbox=tuple(box.xyxy[0].tolist()),
                prompt=f"coco:{cls_name}",
            ))
        return out


# ---------------------------------------------------------------------
# DOG_ATTACK -- same situation as THEFT: no dedicated pretrained model
# exists (searched -- only plain dog/cat *presence* detectors are
# available, nothing trained on "attack" as a behavior). A dog biting/
# lunging at a person is a POSE + PROXIMITY pattern over consecutive
# frames, not a single-frame class. Placeholder: COCO-pretrained
# YOLOv8 already detects "dog" and "person" -- feed both boxes into a
# proximity/overlap rule (dog-person IoU above X for Y consecutive
# frames) rather than expecting a one-shot "attack" detector.
# ---------------------------------------------------------------------
class DogAttackHeuristicDetector:
    COCO_CLASSES = {"person", "dog"}

    def __init__(self, conf_threshold: float = 0.4):
        from ultralytics import YOLO
        self.model = YOLO("yolov8n.pt")  # same COCO weights as TheftHeuristicDetector -- reuse one instance if loading both
        self.conf_threshold = conf_threshold

    def detect(self, frame) -> list[Detection]:
        """Same caveat as TheftHeuristicDetector: this returns raw
        person/dog boxes tagged 'DOG_ATTACK_SIGNAL', not an alert-ready
        category. Needs a proximity-over-time rule layered on top
        before it becomes a real DOG_ATTACK alert."""
        results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
        out = []
        for box in results.boxes:
            cls_name = results.names[int(box.cls)]
            if cls_name not in self.COCO_CLASSES:
                continue
            out.append(Detection(
                category="DOG_ATTACK_SIGNAL",
                confidence=float(box.conf),
                bbox=tuple(box.xyxy[0].tolist()),
                prompt=f"coco:{cls_name}",
            ))
        return out


def build_registry(roboflow_api_key: str | None = None) -> dict:
    """Returns {category_group: detector_instance}. Call .detect(frame) on
    each per sampled frame in main.py's loop, concatenate the Detection
    lists, feed into temporal.py exactly like today's single-model output.

    WeaponDetector is intentionally NOT instantiated here -- see its
    class docstring above. Zero-shot's own "firearm" prompt already
    covers WEAPON correctly; re-add this once a validated replacement
    model is found and confirmed via debug_weapon.py.
    """
    registry = {
        "fire_safety": FireSmokeDetectorV2(),
        "fight": FightDetector(),
        # "weapon": WeaponDetector(),  # DISABLED -- see class docstring, batch_test.py evidence
        # Threat-Detection-YOLOv8n also tested and rejected -- see its
        # class docstring below. Both weapon models from this author
        # failed the same way (high-confidence false positives, same or
        # higher than true positives). Zero-shot's own "firearm" prompt
        # remains the only reliable WEAPON source for now.
    }
    if roboflow_api_key:
        registry["electrical_arc"] = ElectricalArcDetector(api_key=roboflow_api_key)
        # road-accident-qqade is a PUBLIC Roboflow Universe project, so
        # the free "Public" plan API key works here -- no private
        # workspace needed for this specific model.
        registry["accident"] = AccidentDetector(api_key=roboflow_api_key)
    # THEFT and DOG_ATTACK share one COCO model instance -- both are
    # heuristic-signal-only, not alert-ready (see class docstrings).
    coco_shared = TheftHeuristicDetector()
    registry["theft_heuristic"] = coco_shared
    registry["dog_attack_heuristic"] = DogAttackHeuristicDetector()  # separate instance for now; merge if you want to share weights in memory
    return registry