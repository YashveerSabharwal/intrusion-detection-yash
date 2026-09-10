"""
main.py
CCTV incident detection prototype (V1.2) -- YOLO-World + prompts + small
dedicated pretrained models + temporal confirmation, CPU only.

Every run also produces output/report.html: a self-contained analysis
page showing, per category, the highest confidence the pipeline actually
reached (even below your threshold), whether it was confirmed, and
thumbnail crops of the best moments -- useful when something you expect
to be flagged isn't (or something you DON'T expect to be flagged is).

Usage:
    python main.py --input input/test.mp4
    python main.py --input input/test.mp4 --output output/result.mp4
    python main.py --input input/test.mp4 --max-frames 100
    python main.py --input input/test.mp4 --display
"""

import argparse
import os
import sys
import time

import cv2

import config
from detector import IncidentDetector
from temporal import TemporalTracker
from alerts import AlertEngine
from video import VideoSource, VideoSink, draw_detections
from report import build_report, encode_thumbnail


def parse_args():
    parser = argparse.ArgumentParser(description="CCTV incident detection prototype (YOLO-World, CPU-only).")
    parser.add_argument("--input", default=config.INPUT_PATH, help="Path to input MP4 video.")
    parser.add_argument("--output", default=config.OUTPUT_PATH, help="Path to write annotated MP4.")
    parser.add_argument("--max-frames", type=int, default=None, help="Only process the first N raw frames (quick test).")
    parser.add_argument("--display", action="store_true", help="Show annotated video while processing.")
    return parser.parse_args()


def print_banner(input_path, process_fps):
    print("=" * 40)
    print("CCTV INCIDENT DETECTOR")
    print("=" * 40)
    print("Model:  YOLO-World")
    print("Device: CPU")
    print(f"Input:  {input_path}")
    print(f"Processing FPS: {process_fps}")
    print("=" * 40)


def _threshold_for(det):
    """
    FIRE/SMOKE (and potentially WEAPON, once calibrated) can come from
    more than one model source with different confidence scales -- check
    config.SOURCE_CONFIDENCE_THRESHOLDS (matched against the START of
    Detection.prompt) before falling back to the shared per-category
    CONFIDENCE_THRESHOLDS.
    """
    for prefix, thr in config.SOURCE_CONFIDENCE_THRESHOLDS.items():
        if det.prompt and det.prompt.startswith(prefix):
            return thr
    return config.CONFIDENCE_THRESHOLDS.get(det.category, 0.4)


def _suppress_by_confirmed_events(raw_detections):
    """
    Cross-category suppression for suppressors that are REAL alert
    categories (FIRE, SMOKE) rather than suppressor-only prompts like
    ANIMAL (ANIMAL is already handled inside detector.py).

    This has to run HERE, after zero-shot and multi-model (Thalos)
    detections are merged, because a fire might be caught by only ONE of
    the two sources on a given frame -- doing this suppression inside
    detector.py would only ever see zero-shot's own FIRE/SMOKE hits and
    miss a fire that only Thalos detected (or vice versa).

    A FIRE/SMOKE detection only counts as a valid suppressor once it
    clears ITS OWN real alert threshold (via _threshold_for(), the same
    per-source logic used for the main alert pipeline) -- floor-level
    noise shouldn't be able to suppress a detection in another category.

    This is what fixes the car-fire clip where smoke was scoring 0.92+
    as FALLEN_TREE and flame/debris shapes were scoring 0.83+ as WEAPON:
    once FIRE/SMOKE themselves clear threshold in the same frame and
    overlap those boxes, the false positives are dropped.
    """
    to_drop = set()
    for target_category, suppressor_entries in config.SUPPRESSED_BY.items():
        for entry in suppressor_entries:
            suppressor_category = entry["category"]
            if suppressor_category in config.SUPPRESSOR_CATEGORIES:
                continue  # handled already inside detector.py (e.g. ANIMAL)

            iou_threshold = entry["iou"]
            suppressors = [
                d for d in raw_detections
                if d.category == suppressor_category and d.confidence >= _threshold_for(d)
            ]
            if not suppressors:
                continue

            for i, det in enumerate(raw_detections):
                if det.category != target_category or i in to_drop:
                    continue
                for sup in suppressors:
                    if IncidentDetector._iou(det.bbox, sup.bbox) >= iou_threshold:
                        to_drop.add(i)
                        break
    return [d for i, d in enumerate(raw_detections) if i not in to_drop]


def main():
    args = parse_args()
    print_banner(args.input, config.PROCESS_FPS)

    try:
        source = VideoSource(args.input)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    try:
        detector = IncidentDetector()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        source.release()
        sys.exit(1)

    # Small dedicated pretrained models -- FIRE/SMOKE/spark (Thalos),
    # FIGHT, WEAPON-firearms, plus THEFT/DOG_ATTACK raw signal boxes
    # (not alert-ready yet, see multi_model_detector.py). Optional --
    # if this fails to load (missing package, no internet for the first
    # download), fall back to YOLO-World-only behavior rather than
    # crashing the whole run.
    try:
        from multi_model_detector import build_registry
        multi_models = build_registry(roboflow_api_key=os.environ.get("ROBOFLOW_API_KEY"))
    except Exception as e:
        print(f"WARNING: multi-model registry failed to load ({e}) -- continuing with YOLO-World only.")
        multi_models = {}

    try:
        sink = VideoSink(args.output, source.fps, source.width, source.height)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        source.release()
        sys.exit(1)

    tracker = TemporalTracker()
    alert_engine = AlertEngine()

    # --- report-tracking state: independent of the alert pipeline ---
    # FIGHT/ELECTRICAL aren't in CATEGORY_PROMPTS (they have no YOLO-World
    # prompts -- they come from multi_model_detector.py instead), so they
    # need to be added explicitly or the report/temporal loop never sees them.
    categories = list(config.CATEGORY_PROMPTS.keys()) + config.MODEL_BASED_CATEGORIES
    report_stats = {
        c: {
            "threshold": config.CONFIDENCE_THRESHOLDS.get(c, 0.4),
            "min_hits": config.MIN_HITS.get(c, 3),
            "window": config.TEMPORAL_WINDOW,
            "max_confidence": 0.0,
            "confirmed": False,
            "timeline": [],          # (frame_idx, confidence) per sampled frame
            "top_detections": [],    # (frame_idx, timestamp, confidence, thumb_b64, source)
        }
        for c in categories
    }

    # Sample ~PROCESS_FPS frames per second of video.
    sample_every_n = max(1, round(source.fps / config.PROCESS_FPS))

    frames_processed = 0     # raw frames written to output
    frames_inferred = 0      # frames actually run through YOLO-World
    inference_time_total = 0.0
    last_detections = []
    last_confirmed = {c: False for c in categories}

    start_time = time.time()

    try:
        for frame_idx, frame in source.frames(max_frames=args.max_frames):
            if frame_idx % sample_every_n == 0:
                t0 = time.time()
                try:
                    raw_detections = detector.infer_raw(frame)
                except Exception as e:
                    print(f"WARNING: inference failed on frame {frame_idx}: {e}")
                    raw_detections = []

                # Run the small dedicated models on the same sampled frame
                # and merge their output in. THEFT_SIGNAL/DOG_ATTACK_SIGNAL
                # are dropped here -- they're raw person/dog/bag boxes, not
                # alert-ready categories, and must never reach the temporal
                # tracker (see config.SIGNAL_ONLY_CATEGORIES).
                multi_detections = []
                for model_name, model in multi_models.items():
                    try:
                        dets = model.detect(frame)
                    except Exception as e:
                        print(f"WARNING: {model_name} inference failed on frame {frame_idx}: {e}")
                        dets = []
                    multi_detections.extend(
                        d for d in dets if d.category not in config.SIGNAL_ONLY_CATEGORIES
                    )
                if multi_detections:
                    # reuse detector.py's own IoU dedupe so two models
                    # firing on the same WEAPON box don't draw/count twice
                    raw_detections = detector._dedupe(raw_detections + multi_detections)

                # Cross-category suppression for FIRE/SMOKE -> WEAPON /
                # FALLEN_TREE. Must run AFTER the merge above so it sees
                # Thalos's FIRE/SMOKE detections too, not just zero-shot's.
                # (ANIMAL -> FALLEN_PERSON suppression already happened
                # inside detector.infer_raw(), since ANIMAL has no
                # multi-model equivalent.)
                raw_detections = _suppress_by_confirmed_events(raw_detections)

                inference_time_total += time.time() - t0
                frames_inferred += 1

                # best raw detection per category this frame (for report + timeline)
                best_raw_this_frame = {}
                for det in raw_detections:
                    cur = best_raw_this_frame.get(det.category)
                    if cur is None or det.confidence > cur.confidence:
                        best_raw_this_frame[det.category] = det

                timestamp = f"{int(frame_idx / source.fps // 60):02d}:{int(frame_idx / source.fps % 60):02d}"
                for c in categories:
                    det = best_raw_this_frame.get(c)
                    conf = det.confidence if det else 0.0
                    report_stats[c]["timeline"].append((frame_idx, conf))
                    if det and conf > report_stats[c]["max_confidence"]:
                        report_stats[c]["max_confidence"] = conf
                    if det:
                        thumbs = report_stats[c]["top_detections"]
                        thumb_b64 = encode_thumbnail(frame, det.bbox)
                        thumbs.append((frame_idx, timestamp, conf, thumb_b64, det.prompt))
                        thumbs.sort(key=lambda t: t[2], reverse=True)
                        del thumbs[config.REPORT_TOP_K_PER_CATEGORY:]

                # filtered detections (what the real pipeline uses)
                detections = [
                    det for det in raw_detections
                    if det.confidence >= _threshold_for(det)
                ]

                confirmed = tracker.update(detections)

                for category, is_confirmed in confirmed.items():
                    if is_confirmed:
                        report_stats[category]["confirmed"] = True
                        rep = tracker.get_representative(category)
                        if rep is not None:
                            alert = alert_engine.maybe_alert(category, rep, frame_idx, source.fps)
                            if alert:
                                print(f"ALERT: {alert['event']} at {alert['timestamp']} "
                                      f"(conf={alert['confidence']}, source={alert['source']}, frame={alert['frame']})")

                last_detections = detections
                last_confirmed = confirmed

            annotated = draw_detections(frame.copy(), last_detections, last_confirmed)
            sink.write(annotated)
            frames_processed += 1

            if args.display:
                try:
                    cv2.imshow("CCTV Incident Detector", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    # Headless environment with no GUI support -- keep processing.
                    pass

    finally:
        source.release()
        sink.release()
        if args.display:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    alerts_path = alert_engine.save(config.ALERTS_PATH)
    total_time = time.time() - start_time
    actual_fps = frames_inferred / inference_time_total if inference_time_total > 0 else 0.0

    video_info = {
        "input": args.input,
        "fps": source.fps,
        "width": source.width,
        "height": source.height,
        "frame_count": source.frame_count,
        "frames_processed": frames_processed,
        "frames_inferred": frames_inferred,
        "total_time": total_time,
    }
    report_path = build_report(config.REPORT_PATH, video_info, report_stats, alert_engine.alerts)

    print("=" * 40)
    print(f"Frames processed:  {frames_processed}")
    print(f"Frames inferred:   {frames_inferred}")
    print(f"Inference time:    {inference_time_total:.2f}s")
    print(f"Processing FPS:    {actual_fps:.2f}")
    print(f"Total time:        {total_time:.2f}s")
    print(f"Annotated video:   {args.output}")
    print(f"Alerts JSON:       {alerts_path} ({len(alert_engine.alerts)} alerts)")
    print(f"Analysis report:   {report_path}")
    print("=" * 40)


if __name__ == "__main__":
    main()