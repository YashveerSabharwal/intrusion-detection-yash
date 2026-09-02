"""
temporal.py
Tracks recent per-category hit/miss history across sampled frames and
decides when a category is "confirmed" (i.e. persistent enough to alert on).
No single-frame detection ever triggers an alert on its own.
"""

from collections import deque

import config


class TemporalTracker:
    def __init__(self, window=None, min_hits=None):
        self.window = window or config.TEMPORAL_WINDOW
        self.min_hits = min_hits or config.MIN_HITS
        self.history = {
            category: deque(maxlen=self.window) for category in config.CATEGORY_PROMPTS
        }
        # best detection seen in the current window, per category
        # (used so the alert/overlay has a representative bbox+confidence)
        self.best_in_window = {category: None for category in config.CATEGORY_PROMPTS}

    def update(self, detections_this_frame):
        """
        detections_this_frame: list of Detection objects from detector.infer()
        Returns dict: {category: confirmed(bool)} for every category.
        """
        present_categories = {}
        for det in detections_this_frame:
            current_best = present_categories.get(det.category)
            if current_best is None or det.confidence > current_best.confidence:
                present_categories[det.category] = det

        confirmed = {}
        for category in self.history:
            hit = category in present_categories
            self.history[category].append(hit)

            if hit:
                self.best_in_window[category] = present_categories[category]

            hits = sum(self.history[category])
            required = self.min_hits.get(category, 3)
            confirmed[category] = hits >= required

            if not hit and hits == 0:
                self.best_in_window[category] = None

        return confirmed

    def get_representative(self, category):
        """Best detection currently backing up a confirmed category."""
        return self.best_in_window.get(category)