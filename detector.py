"""
detector.py
Thin wrapper around YOLO-World: load model, set open-vocabulary prompts,
run inference on a single frame, and map raw prompt hits back to the
5 high-level incident categories.
"""

import config


class Detection:
    """One filtered detection, already mapped to a high-level category."""

    __slots__ = ("category", "confidence", "bbox", "prompt")

    def __init__(self, category, confidence, bbox, prompt):
        self.category = category          # e.g. "FIRE"
        self.confidence = confidence       # float 0-1
        self.bbox = bbox                   # [x1, y1, x2, y2] ints
        self.prompt = prompt               # raw matched prompt, e.g. "flames"

    def as_dict(self):
        return {
            "category": self.category,
            "confidence": round(float(self.confidence), 3),
            "bbox": [int(v) for v in self.bbox],
            "prompt": self.prompt,
        }


class IncidentDetector:
    def __init__(self, model_path=None, device=None):
        self.model_path = model_path or config.MODEL_PATH
        self.device = device or config.DEVICE
        self.model = self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError(
                "ultralytics is not installed. Run: pip install ultralytics"
            ) from e

        try:
            model = YOLO(self.model_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load YOLO-World checkpoint '{self.model_path}'. "
                f"Ultralytics should auto-download it on first use if your "
                f"internet connection allows it. Original error: {e}"
            ) from e

        # Open-vocabulary prompt list -> model's active classes.
        try:
            model.set_classes(config.ALL_PROMPTS)
        except Exception as e:
            raise RuntimeError(
                "model.set_classes() failed. Your installed ultralytics "
                "version may use a different YOLO-World API. Check "
                "`pip show ultralytics` and the current Ultralytics docs "
                f"for YOLO-World custom classes. Original error: {e}"
            ) from e

        return model

    def infer_raw(self, frame):
        """
        Run YOLO-World on a single BGR frame (numpy array) and return
        EVERY detection above config.RAW_CONFIDENCE_FLOOR, mapped to a
        high-level category, with NO per-category confidence filtering
        applied yet. Use this for debugging/reporting -- it shows you
        everything the model actually saw, including things that would
        get thrown away by CONFIDENCE_THRESHOLDS.
        """
        results = self.model.predict(
            frame,
            device=self.device,
            conf=config.RAW_CONFIDENCE_FLOOR,
            verbose=False,
        )

        raw = []
        if not results:
            return raw

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return raw

        names = result.names  # id -> prompt string, as set by set_classes()

        for box in result.boxes:
            cls_id = int(box.cls[0])
            prompt = names.get(cls_id, None) if isinstance(names, dict) else names[cls_id]
            category = config.PROMPT_TO_CATEGORY.get(prompt)
            if category is None:
                continue  # shouldn't happen, but stay safe

            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()

            # Geometric sanity filter: for categories that describe an
            # ORIENTATION (fallen == lying down / horizontal), reject
            # boxes shaped like a standing object. Confidence alone
            # can't be trusted for this -- on real footage a standing
            # person scored HIGHER (0.806) than a genuinely fallen one
            # (0.428). See config.MIN_ASPECT_RATIO for details.
            min_ratio = config.MIN_ASPECT_RATIO.get(category)
            if min_ratio is not None:
                x1, y1, x2, y2 = xyxy
                width, height = x2 - x1, y2 - y1
                if height <= 0 or (width / height) < min_ratio:
                    continue  # too tall/narrow to be a "fallen" orientation

            raw.append(Detection(category, conf, xyxy, prompt))

        raw = self._dedupe(raw)
        raw = self._suppress_cross_category(raw)
        # Suppressor categories (e.g. ANIMAL) exist only to cross-check
        # other categories above -- they're not real alert categories,
        # so strip them out before anything downstream (temporal
        # tracker, alerts, report) ever sees them.
        return [d for d in raw if d.category not in config.SUPPRESSOR_CATEGORIES]

    def infer(self, frame):
        """
        Same as infer_raw(), but filtered down to detections that clear
        each category's CONFIDENCE_THRESHOLDS. This is what the main
        pipeline (temporal confirmation / alerts) actually uses.
        """
        raw = self.infer_raw(frame)
        return [
            det for det in raw
            if det.confidence >= config.CONFIDENCE_THRESHOLDS.get(det.category, 0.4)
        ]

    @staticmethod
    def _iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _suppress_cross_category(self, detections):
        """
        Some real-world objects are geometrically indistinguishable from
        an alert category at the bounding-box level -- e.g. a standing
        dog's box is the same wide/short shape as a fallen person's box,
        so MIN_ASPECT_RATIO can't separate them. For each (category,
        suppressor) pair in config.SUPPRESSED_BY, drop any detection in
        `category` that overlaps a same-frame detection in `suppressor`
        by at least the configured IoU threshold.
        """
        to_drop = set()
        for target_category, (suppressor_category, iou_threshold) in config.SUPPRESSED_BY.items():
            suppressors = [d for d in detections if d.category == suppressor_category]
            if not suppressors:
                continue
            for i, det in enumerate(detections):
                if det.category != target_category or i in to_drop:
                    continue
                for sup in suppressors:
                    if self._iou(det.bbox, sup.bbox) >= iou_threshold:
                        to_drop.add(i)
                        break
        return [d for i, d in enumerate(detections) if i not in to_drop]

    def _dedupe(self, detections, iou_threshold=0.5):
        """
        Multiple prompts in the same category (e.g. 'gun' and 'pistol')
        can fire on the same object. Keep only the highest-confidence
        box per overlapping cluster, within the same category.
        """
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
        kept = []
        for det in detections:
            duplicate = False
            for k in kept:
                if k.category == det.category and self._iou(k.bbox, det.bbox) > iou_threshold:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(det)
        return kept