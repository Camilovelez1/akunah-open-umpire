"""Table geometry: where the two halves and the net are in the image.

Geometry is stored as JSON in pixel coordinates of a given frame size and is
rescaled automatically if the video resolution changes.
"""

from __future__ import annotations

import json
from itertools import combinations
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .rules import Side

# Ratio of net height (15.25 cm) to table width (152.5 cm).
NET_HEIGHT_RATIO = 0.1


@dataclass
class TableGeometry:
    # Corners of the playing surface, ordered: left-near, right-near, right-far, left-far
    # ("near"/"far" only matter for drawing; left/right are image left/right).
    corners: np.ndarray
    frame_size: tuple[int, int]  # (width, height)

    @property
    def homography(self) -> np.ndarray:
        """Maps table coordinates (u along the length 0..1, v across 0..1) to image pixels."""
        unit = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
        return cv2.getPerspectiveTransform(unit, self.corners.astype(np.float32))

    def to_image(self, uv: np.ndarray) -> np.ndarray:
        pts = np.asarray(uv, np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.homography).reshape(-1, 2)

    def half(self, side: Side) -> np.ndarray:
        u0, u1 = (0.0, 0.5) if side is Side.LEFT else (0.5, 1.0)
        return self.to_image([[u0, 0], [u1, 0], [u1, 1], [u0, 1]])

    @property
    def net_base(self) -> np.ndarray:
        return self.to_image([[0.5, 0], [0.5, 1]])

    def net(self) -> np.ndarray:
        """Net as an image polygon: the centre line raised by the net height."""
        base = self.net_base
        width_px = float(np.linalg.norm(base[1] - base[0]))
        lift = np.array([0, -NET_HEIGHT_RATIO * width_px * 1.2], np.float32)
        return np.array([base[0], base[1], base[1] + lift, base[0] + lift])

    def side_at(self, point: tuple[float, float], margin: float = 6.0) -> Side | None:
        """Half of the table under `point`, accepting points up to `margin` px outside it.

        A ball bouncing is seen slightly above the surface in a side view, so a
        small vertical tolerance matters more than precision.
        """
        best, best_dist = None, -margin
        for side in Side:
            dist = cv2.pointPolygonTest(self.half(side).astype(np.float32), tuple(map(float, point)), True)
            if dist >= best_dist:
                best, best_dist = side, dist
        return best

    def scaled_to(self, frame_size: tuple[int, int]) -> TableGeometry:
        if tuple(frame_size) == tuple(self.frame_size):
            return self
        sx = frame_size[0] / self.frame_size[0]
        sy = frame_size[1] / self.frame_size[1]
        return TableGeometry(self.corners * np.float32([sx, sy]), tuple(frame_size))

    def draw(self, frame: np.ndarray) -> None:
        colours = {Side.LEFT: (255, 160, 0), Side.RIGHT: (0, 160, 255)}
        for side in Side:
            cv2.polylines(frame, [self.half(side).astype(np.int32)], True, colours[side], 2)
        cv2.polylines(frame, [self.net().astype(np.int32)], True, (255, 255, 255), 1)

    def save(self, path: str | Path) -> None:
        data = {"frame_size": list(self.frame_size), "corners": self.corners.round(1).tolist()}
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> TableGeometry:
        data = json.loads(Path(path).read_text())
        return cls(np.float32(data["corners"]), tuple(data["frame_size"]))

    @classmethod
    def from_corners(cls, corners, frame_size: tuple[int, int]) -> TableGeometry:
        return cls(order_corners(np.float32(corners)), tuple(frame_size))


def order_corners(quad: np.ndarray) -> np.ndarray:
    """Order 4 points so that u (corner0 -> corner1) runs along the table length, left to right."""
    quad = np.float32(quad).reshape(4, 2)
    centre = quad.mean(axis=0)
    angles = np.arctan2(quad[:, 1] - centre[1], quad[:, 0] - centre[0])
    quad = quad[np.argsort(angles)]  # clockwise in image coordinates, starting anywhere
    edges = [np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4)]
    # The pair of opposite edges with the larger total length is the table length.
    start = 0 if edges[0] + edges[2] >= edges[1] + edges[3] else 1
    quad = np.roll(quad, -start, axis=0)
    # Now quad[0]->quad[1] and quad[2]->quad[3] are long edges. Make u point left to right.
    if quad[1][0] < quad[0][0]:
        quad = np.roll(quad, 2, axis=0)
    # Put the edge with larger y (nearer the camera) first.
    if quad[0][1] + quad[1][1] < quad[2][1] + quad[3][1]:
        quad = quad[[3, 2, 1, 0]]
    return quad


@dataclass
class Detection:
    geometry: TableGeometry
    mask: np.ndarray
    score: float
    hue: int


def detect_table(
    frames: list[np.ndarray] | np.ndarray,
    seed: tuple[int, int] | None = None,
    hue_range: tuple[int, int] = (35, 135),
) -> Detection | None:
    """Find the playing surface by colour.

    Several frames are median-blended first so players and the ball disappear.
    Without a seed point, every hue window in `hue_range` is tried and the blob
    that looks most like a table top (big, solid, four-sided, wider than tall)
    wins. With a seed, the hue under the seed is used and the blob containing
    it is taken.
    """
    if isinstance(frames, np.ndarray) and frames.ndim == 3:
        frames = [frames]
    frame = np.median(np.stack(frames), axis=0).astype(np.uint8) if len(frames) > 1 else frames[0]
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)

    if seed is not None:
        patch = hsv[max(seed[1] - 3, 0) : seed[1] + 4, max(seed[0] - 3, 0) : seed[0] + 4].reshape(-1, 3)
        hues = [int(np.median(patch[:, 0]))]
    else:
        hues = list(range(hue_range[0], hue_range[1] + 1, 2))

    best: Detection | None = None
    for hue in hues:
        mask = _hue_mask(hsv, hue)
        for contour in _candidate_contours(mask, w, h):
            if seed is not None and cv2.pointPolygonTest(contour, (float(seed[0]), float(seed[1])), False) < 0:
                continue
            blob = np.zeros_like(mask)
            cv2.drawContours(blob, [contour], -1, 255, -1)
            # Share of the blob that really has the colour: a plain table top is
            # nearly solid, striped backgrounds (bleachers, walls) only look solid after closing.
            fill = cv2.countNonZero(cv2.bitwise_and(mask, blob)) / max(cv2.countNonZero(blob), 1)
            quad, score = _table_score(contour, w, h)
            saturation = float(hsv[..., 1][blob > 0].mean()) / 128
            score *= fill**4 * saturation
            if quad is not None and (best is None or score > best.score):
                best = Detection(TableGeometry.from_corners(quad, (w, h)), blob, score, hue)
    if best is not None:
        contour = max(cv2.findContours(best.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
        refined = _refine_quad(best.geometry.corners, cv2.convexHull(contour))
        best.geometry = TableGeometry.from_corners(refined, (w, h))
    return best


def _refine_quad(quad: np.ndarray, hull: np.ndarray) -> np.ndarray:
    """Fit a line to each side of the hull and intersect neighbours.

    Seen from a low angle, the ends of the table are thin wedges that the mask
    loses, so hull vertices sit inside the real corners. The long edges are
    well defined, and extending them recovers the corners.
    """
    poly = hull.reshape(-1, 2).astype(np.float32)
    samples = []
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        n = max(int(np.linalg.norm(b - a) // 2), 1)
        samples.append(a + (b - a) * np.linspace(0, 1, n, endpoint=False)[:, None])
    pts = np.concatenate(samples)
    edges = [(quad[i], quad[(i + 1) % 4]) for i in range(4)]
    dists = np.stack([_point_segment_distance(pts, a, b) for a, b in edges], axis=1)
    owner = dists.argmin(axis=1)
    lines = []
    for i, (a, b) in enumerate(edges):
        mine = pts[(owner == i) & (dists[:, i] < 4)]
        if len(mine) < 5:
            lines.append((a, b - a))
            continue
        vx, vy, x0, y0 = cv2.fitLine(mine, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.float32([x0, y0]), np.float32([vx, vy])))
    corners = []
    for i in range(4):
        p = _intersect(lines[(i - 1) % 4], lines[i])
        corners.append(quad[i] if p is None or np.linalg.norm(p - quad[i]) > 0.25 * np.ptp(quad[:, 0]) else p)
    return np.float32(corners)


def _point_segment_distance(pts: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    t = np.clip(((pts - a) @ ab) / max(float(ab @ ab), 1e-9), 0, 1)
    return np.linalg.norm(pts - (a + t[:, None] * ab), axis=1)


def _intersect(l1, l2) -> np.ndarray | None:
    (p, r), (q, s) = l1, l2
    denom = r[0] * s[1] - r[1] * s[0]
    if abs(denom) < 1e-6:
        return None
    t = ((q[0] - p[0]) * s[1] - (q[1] - p[1]) * s[0]) / denom
    return np.float32(p + t * r)


def _hue_mask(hsv: np.ndarray, hue: int, tolerance: int = 4) -> np.ndarray:
    # Table paint is clearly saturated; greyish walls and floors are not.
    lo = np.array([max(hue - tolerance, 0), 70, 40], np.uint8)
    hi = np.array([min(hue + tolerance, 179), 255, 255], np.uint8)
    return cv2.inRange(hsv, lo, hi)


def _candidate_contours(mask: np.ndarray, w: int, h: int) -> list[np.ndarray]:
    # Closing bridges the white centre line and the net so the top becomes one blob.
    k = max(3, int(round(w * 0.012)) | 1)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 0.005 * w * h
    return [c for c in contours if cv2.contourArea(c) >= min_area]


def _table_score(contour: np.ndarray, w: int, h: int) -> tuple[np.ndarray | None, float]:
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        return None, 0.0
    quad = _approx_quad(hull)
    quad_area = cv2.contourArea(quad.astype(np.float32))
    if quad_area <= 0:
        return None, 0.0
    solidity = area / hull_area
    # How well the blob fills its 4-sided approximation (and vice versa).
    fit = min(area, quad_area) / max(area, quad_area)
    x, y, bw, bh = cv2.boundingRect(contour)
    if bw < 1.5 * bh:
        return None, 0.0  # a table top seen from the side is always wider than tall
    # Banners and walls touch the image border; a table usually doesn't.
    border = 0.6 if x <= 1 or x + bw >= w - 1 else 1.0
    score = (area / (w * h)) * solidity * fit * border
    return quad, score


def _approx_quad(hull: np.ndarray) -> np.ndarray:
    """Largest-area quadrilateral whose corners are vertices of the convex hull."""
    pts = hull.reshape(-1, 2).astype(np.float64)
    if len(pts) > 40:
        perimeter = cv2.arcLength(hull, True)
        pts = cv2.approxPolyDP(hull, 0.002 * perimeter, True).reshape(-1, 2).astype(np.float64)
    if len(pts) < 4:
        return cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)
    # Hull vertices are in order, so any 4 of them taken in index order form a convex quad.
    idx = np.array(list(combinations(range(len(pts)), 4)))
    q = pts[idx]  # (n, 4, 2)
    x, y = q[..., 0], q[..., 1]
    area = 0.5 * np.abs((x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y).sum(axis=1))
    return q[int(np.argmax(area))].astype(np.float32)
