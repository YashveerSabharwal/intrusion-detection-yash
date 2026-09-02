"""
main.py
CCTV incident detection prototype (V1) -- YOLO-World + prompts + temporal
confirmation, CPU only. No training, no tracking, no VLM.

Every run also produces output/report.html: a self-contained analysis
page showing, per category, the highest confidence YOLO-World actually
reached (even below your threshold), whether it was confirmed, and
thumbnail crops of the best moments -- useful when something you expect
to be flagged isn't.

Usage:
    python main.py --input input/test.mp4
    python main.py --input input/test.mp4 --output output/result.mp4
    python main.py --input input/test.mp4 --max-frames 100
    python main.py --input input/test.mp4 --display
"""

import argparse
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

    try:
        sink = VideoSink(args.output, source.fps, source.width, source.height)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        source.release()
        sys.exit(1)

    tracker = TemporalTracker()
    alert_engine = AlertEngine()

    # --- report-tracking state: independent of the alert pipeline ---
    categories = list(config.CATEGORY_PROMPTS.keys())
    report_stats = {
        c: {
            "threshold": config.CONFIDENCE_THRESHOLDS.get(c, 0.4),
            "min_hits": config.MIN_HITS.get(c, 3),
            "window": config.TEMPORAL_WINDOW,
            "max_confidence": 0.0,
            "confirmed": False,
            "timeline": [],          # (frame_idx, confidence) per sampled frame
            "top_detections": [],    # (frame_idx, timestamp, confidence, thumb_b64)
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
                        thumbs.append((frame_idx, timestamp, conf, thumb_b64))
                        thumbs.sort(key=lambda t: t[2], reverse=True)
                        del thumbs[config.REPORT_TOP_K_PER_CATEGORY:]

                # filtered detections (what the real pipeline uses)
                detections = [
                    det for det in raw_detections
                    if det.confidence >= config.CONFIDENCE_THRESHOLDS.get(det.category, 0.4)
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
                                      f"(conf={alert['confidence']}, frame={alert['frame']})")

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