"""
report.py
Builds a single, self-contained output/report.html after every run.
Shows, per category: whether it was ever seen at all, the highest
confidence reached, whether it cleared the threshold, whether it was
temporally confirmed, a confidence sparkline over time, and thumbnail
crops of the best moments -- so you can tell "model never saw it" apart
from "saw it but threshold/temporal logic filtered it out".
"""

import base64
import os

import cv2

import config


CATEGORY_COLORS = {
    "FIRE": "#ff4500",
    "SMOKE": "#9e9e9e",
    "WEAPON": "#e53935",
    "FALLEN_PERSON": "#fbc02d",
    "FALLEN_TREE": "#8b5a2b",
    "FIGHT": "#fb8c00",
    "ELECTRICAL": "#00acc1",
}


def encode_thumbnail(frame, bbox=None, max_width=360):
    img = frame.copy()
    if bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (max_width, int(h * scale)))
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
    if not ok:
        return ""
    return base64.b64encode(buf).decode("utf-8")


def _sparkline_svg(points, threshold, color, width=600, height=80):
    """points: list of (frame_idx, confidence) for sampled frames, in order."""
    if not points:
        return "<svg></svg>"

    max_conf = max(1.0, max(c for _, c in points))
    n = len(points)
    step = width / max(1, n - 1)

    coords = []
    for i, (_, conf) in enumerate(points):
        x = i * step
        y = height - (conf / max_conf) * height
        coords.append(f"{x:.1f},{y:.1f}")
    polyline = " ".join(coords)

    threshold_y = height - (threshold / max_conf) * height

    return f"""
<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" xmlns="http://www.w3.org/2000/svg">
  <line x1="0" y1="{threshold_y:.1f}" x2="{width}" y2="{threshold_y:.1f}"
        stroke="#666" stroke-dasharray="4,3" stroke-width="1" />
  <polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2" />
</svg>
"""


def build_report(output_path, video_info, category_data, alerts):
    """
    video_info: dict with keys input, fps, width, height, frame_count,
                frames_processed, frames_inferred, total_time
    category_data: dict category -> {
        "threshold": float,
        "min_hits": int,
        "window": int,
        "max_confidence": float,
        "confirmed": bool,
        "timeline": [(frame_idx, confidence), ...],   # 0.0 when absent
        "top_detections": [(frame_idx, timestamp, confidence, thumb_b64, source), ...]
    }
    alerts: list of alert dicts (from alerts.json)
    """
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    sections = []
    for category, data in category_data.items():
        color = CATEGORY_COLORS.get(category, "#333")
        seen_at_all = data["max_confidence"] > 0.0
        cleared_threshold = data["max_confidence"] >= data["threshold"]

        if data["confirmed"]:
            verdict = ("FLAGGED", "#1b5e20", "#c8e6c9")
        elif cleared_threshold:
            verdict = ("SEEN, BUT NOT PERSISTENT ENOUGH", "#e65100", "#ffe0b2")
        elif seen_at_all:
            verdict = ("SEEN, BELOW THRESHOLD", "#e65100", "#ffe0b2")
        else:
            verdict = ("NEVER DETECTED", "#b71c1c", "#ffcdd2")

        verdict_label, verdict_color, verdict_bg = verdict

        spark = _sparkline_svg(data["timeline"], data["threshold"], color)

        thumbs_html = ""
        for entry in data["top_detections"]:
            # Tolerate both the old 4-tuple shape and the new 5-tuple
            # shape (adds `source`), in case older report_stats data is
            # ever passed in.
            if len(entry) == 5:
                frame_idx, ts, conf, thumb, source = entry
            else:
                frame_idx, ts, conf, thumb = entry
                source = None

            if not thumb:
                continue

            source_html = f" &middot; <span class='source'>{source}</span>" if source else ""
            thumbs_html += f"""
            <div class="thumb">
              <img src="data:image/jpeg;base64,{thumb}" />
              <div class="thumb-caption">{ts} &middot; conf {conf:.2f} &middot; frame {frame_idx}{source_html}</div>
            </div>"""

        if not thumbs_html:
            thumbs_html = '<div class="no-thumbs">No detections above the raw floor for this category.</div>'

        sections.append(f"""
        <section class="card">
          <div class="card-header">
            <h2 style="color:{color}">{category.replace('_', ' ')}</h2>
            <span class="verdict" style="color:{verdict_color}; background:{verdict_bg}">{verdict_label}</span>
          </div>
          <div class="stats">
            <div><b>Max confidence seen:</b> {data['max_confidence']:.3f}</div>
            <div><b>Threshold:</b> {data['threshold']:.2f}</div>
            <div><b>Temporal rule:</b> {data['min_hits']} hits within last {data['window']} sampled frames</div>
          </div>
          <div class="spark-wrap">{spark}</div>
          <div class="thumbs">{thumbs_html}</div>
        </section>
        """)

    alerts_rows = ""
    if alerts:
        for a in alerts:
            color = CATEGORY_COLORS.get(a["event"], "#333")
            source = a.get("source", "")
            alerts_rows += f"""
            <tr>
              <td style="color:{color}"><b>{a['event']}</b></td>
              <td>{a['timestamp']}</td>
              <td>{a['confidence']:.3f}</td>
              <td>{a['frame']}</td>
              <td>{a['bbox']}</td>
              <td>{source}</td>
            </tr>"""
        alerts_table = f"""
        <table>
          <thead><tr><th>Event</th><th>Timestamp</th><th>Confidence</th><th>Frame</th><th>BBox</th><th>Source</th></tr></thead>
          <tbody>{alerts_rows}</tbody>
        </table>
        """
    else:
        alerts_table = '<p class="none">No alerts were confirmed for this video.</p>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CCTV Incident Analysis Report</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; background:#f4f4f6; color:#1a1a1a; margin:0; padding:24px; }}
  h1 {{ margin-bottom: 4px; }}
  .subtitle {{ color:#666; margin-top:0; margin-bottom: 24px; }}
  .meta {{ background:#fff; border-radius:10px; padding:16px 20px; margin-bottom:24px; box-shadow:0 1px 3px rgba(0,0,0,0.08); }}
  .meta table {{ border-collapse: collapse; }}
  .meta td {{ padding: 2px 12px 2px 0; font-size: 14px; color:#333; }}
  .card {{ background:#fff; border-radius:10px; padding:18px 22px; margin-bottom:20px; box-shadow:0 1px 3px rgba(0,0,0,0.08); }}
  .card-header {{ display:flex; align-items:center; justify-content:space-between; }}
  .card-header h2 {{ margin: 0 0 4px 0; font-size: 20px; }}
  .verdict {{ font-size:12px; font-weight:700; padding:4px 10px; border-radius:999px; letter-spacing:0.03em; }}
  .stats {{ display:flex; gap:24px; font-size:13px; color:#444; margin: 8px 0 12px 0; flex-wrap:wrap; }}
  .spark-wrap {{ background:#fafafa; border-radius:6px; padding:8px; margin-bottom:12px; }}
  .thumbs {{ display:flex; gap:12px; flex-wrap:wrap; }}
  .thumb img {{ display:block; border-radius:6px; max-width:220px; }}
  .thumb-caption {{ font-size:11px; color:#666; margin-top:4px; text-align:center; }}
  .thumb-caption .source {{ font-style:italic; color:#888; }}
  .no-thumbs {{ font-size:13px; color:#999; font-style:italic; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th, td {{ text-align:left; padding:8px 10px; border-bottom:1px solid #eee; }}
  th {{ color:#666; font-weight:600; }}
  .none {{ color:#999; font-style:italic; }}
  .section-title {{ margin-top: 32px; margin-bottom: 10px; }}
</style>
</head>
<body>
  <h1>CCTV Incident Analysis Report</h1>
  <p class="subtitle">YOLO-World + dedicated model prototype &middot; generated automatically after each run</p>

  <div class="meta">
    <table>
      <tr><td><b>Input</b></td><td>{video_info['input']}</td></tr>
      <tr><td><b>Resolution</b></td><td>{video_info['width']}x{video_info['height']}</td></tr>
      <tr><td><b>Video FPS</b></td><td>{video_info['fps']:.2f}</td></tr>
      <tr><td><b>Frames processed / inferred</b></td><td>{video_info['frames_processed']} / {video_info['frames_inferred']}</td></tr>
      <tr><td><b>Total run time</b></td><td>{video_info['total_time']:.2f}s</td></tr>
    </table>
  </div>

  <h2 class="section-title">Confirmed alerts</h2>
  {alerts_table}

  <h2 class="section-title">Per-category analysis</h2>
  {''.join(sections)}

</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_path