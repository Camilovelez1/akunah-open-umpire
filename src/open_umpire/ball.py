"""Ball detection and bounce detection from the image."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from .rules import Side
from .table import TableGeometry

# Table length / ball diameter (274 cm / 4 cm).
TABLE_TO_BALL = 68.5

COLOUR_RANGES = {
    # (lower HSV, upper HSV)
    "orange": ((5, 120, 120), (25, 255, 255)),
    "white": ((0, 0, 190), (179, 60, 255)),
}


@dataclass
class BallObservation:
    t: float
    x: float
    y: float
    radius: float
    score: float


class BallDetector:
    """Finds the ball as a small, round, moving blob near where it is expected.

    Motion comes from a three-frame difference (the object has to differ from
    both previous frames), which unlike a plain subtraction works for a ball
    that is darker or brighter than the background. An optional colour mask
    narrows candidates further.
    """

    def __init__(self, table: TableGeometry | None = None, colour: str | None = "orange", motion_threshold: int = 25):
        self.table = table
        self.colour = colour
        self.motion_threshold = motion_threshold
        self._prev: deque[np.ndarray] = deque(maxlen=2)
        self._last: BallObservation | None = None
        self._velocity = np.zeros(2)
        self._roi: np.ndarray | None = None
        self.debug_mask: np.ndarray | None = None

    def roi(self, shape: tuple[int, ...]) -> np.ndarray | None:
        """Where the ball can matter: above and around the table, not the players' bodies beside it."""
        if self.table is None:
            return None
        if self._roi is None or self._roi.shape != shape[:2]:
            c = self.table.corners
            length = float(np.ptp(c[:, 0]))
            x0, x1 = c[:, 0].min() - 0.08 * length, c[:, 0].max() + 0.08 * length
            y0, y1 = c[:, 1].min() - 0.35 * length, c[:, 1].max() + 0.03 * length
            self._roi = np.zeros(shape[:2], np.uint8)
            cv2.rectangle(self._roi, (int(x0), int(max(y0, 0))), (int(x1), int(y1)), 255, -1)
        return self._roi

    def expected_radius(self, frame_width: int) -> float:
        if self.table is not None:
            c = self.table.corners
            length = (np.linalg.norm(c[1] - c[0]) + np.linalg.norm(c[2] - c[3])) / 2
            return max(length / TABLE_TO_BALL / 2, 1.5)
        return max(frame_width / 200, 1.5)

    def detect(self, frame: np.ndarray, t: float) -> BallObservation | None:
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (3, 3), 0)
        if len(self._prev) < 2:
            self._prev.append(gray)
            return None
        d1 = cv2.absdiff(gray, self._prev[-1])
        d2 = cv2.absdiff(gray, self._prev[-2])
        self._prev.append(gray)
        motion = cv2.bitwise_and(
            cv2.threshold(d1, self.motion_threshold, 255, cv2.THRESH_BINARY)[1],
            cv2.threshold(d2, self.motion_threshold, 255, cv2.THRESH_BINARY)[1],
        )
        mask = motion
        if self.colour in COLOUR_RANGES:
            lo, hi = COLOUR_RANGES[self.colour]
            colour_mask = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), np.array(lo), np.array(hi))
            # A fast ball is blurred, so let colour count where motion is nearby.
            mask = cv2.bitwise_and(colour_mask, cv2.dilate(motion, np.ones((5, 5), np.uint8)))
        roi = self.roi(frame.shape)
        if roi is not None:
            mask = cv2.bitwise_and(mask, roi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        self.debug_mask = mask

        r = self.expected_radius(frame.shape[1])
        expected_area = np.pi * r * r
        predicted = None
        if self._last is not None and t - self._last.t < 0.25:
            predicted = np.array([self._last.x, self._last.y]) + self._velocity * (t - self._last.t)

        best: BallObservation | None = None
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            # Motion blur stretches the ball, so allow up to several times its area.
            if not (0.25 * expected_area <= area <= 6 * expected_area + 4):
                continue
            (x, y), radius = cv2.minEnclosingCircle(c)
            circularity = area / (np.pi * radius * radius + 1e-6)
            size = np.exp(-abs(np.log((area + 1) / (expected_area + 1))))
            score = (0.4 + circularity) * size
            if predicted is not None:
                dist = float(np.hypot(x - predicted[0], y - predicted[1]))
                score *= np.exp(-dist / (20 * r))
            if best is None or score > best.score:
                best = BallObservation(t, float(x), float(y), float(radius), float(score))

        if best is not None and best.score > 0.05:
            if self._last is not None and 0 < t - self._last.t < 0.25:
                v = (np.array([best.x, best.y]) - [self._last.x, self._last.y]) / (t - self._last.t)
                self._velocity = 0.5 * self._velocity + 0.5 * v
            else:
                self._velocity = np.zeros(2)
            self._last = best
            return best
        return None


@dataclass
class Bounce:
    t: float
    x: float
    y: float
    side: Side


class BounceDetector:
    """Detects bounces as the ball going from moving down to moving up over the table.

    Works for a camera looking at the table from the side (straight or at an
    angle). From straight above the height of the ball can't be seen; use audio.
    """

    def __init__(self, table: TableGeometry, min_drop: float | None = None, min_interval: float = 0.15):
        self.table = table
        self.track: deque[BallObservation] = deque(maxlen=64)
        r = BallDetector(table).expected_radius(table.frame_size[0])
        self.min_drop = min_drop if min_drop is not None else max(0.6 * r, 1.0)
        self.margin = max(3 * r, 6)
        self.min_interval = min_interval
        self._last_bounce_t = float("-inf")

    def update(self, obs: BallObservation | None) -> Bounce | None:
        if obs is None:
            return None
        if self.track and obs.t - self.track[-1].t > 0.3:
            self.track.clear()  # lost the ball for too long; don't link old and new tracks
        self.track.append(obs)
        if len(self.track) < 3:
            return None
        a, b, c = self.track[-3], self.track[-2], self.track[-1]
        # Image y grows downwards, so a bounce is a local maximum of y.
        if b.y - a.y >= self.min_drop and b.y - c.y >= self.min_drop and b.t - self._last_bounce_t >= self.min_interval:
            side = self.table.side_at((b.x, b.y), margin=self.margin)
            if side is not None:
                self._last_bounce_t = b.t
                return Bounce(b.t, b.x, b.y, side)
        return None

    def position_at(self, t: float, tolerance: float = 0.06) -> BallObservation | None:
        """Ball observation closest in time to `t`, used to place an audio bounce."""
        best = min(self.track, key=lambda o: abs(o.t - t), default=None)
        if best is None or abs(best.t - t) > tolerance:
            return None
        return best
