import cv2
import numpy as np

from open_umpire.ball import BallDetector, BallObservation, BounceDetector
from open_umpire.rules import Side
from open_umpire.table import TableGeometry, detect_table

W, H = 640, 360
CORNERS = np.float32([[80, 260], [560, 260], [480, 200], [160, 200]])  # near-left, near-right, far-right, far-left


def scene():
    img = np.full((H, W, 3), (60, 90, 120), np.uint8)  # brownish floor
    cv2.rectangle(img, (0, 0), (W, 120), (90, 90, 95), -1)  # grey wall
    cv2.fillConvexPoly(img, CORNERS.astype(np.int32), (140, 90, 30))  # blue table top (BGR)
    cv2.line(img, (120, 230), (520, 230), (255, 255, 255), 2)  # centre line
    cv2.line(img, (320, 200), (320, 262), (230, 230, 230), 3)  # net
    return img


def test_detects_synthetic_table_corners():
    det = detect_table(scene())
    assert det is not None
    np.testing.assert_allclose(det.geometry.corners, CORNERS, atol=6)


def test_detects_table_from_seed_point():
    det = detect_table(scene(), seed=(200, 240))
    np.testing.assert_allclose(det.geometry.corners, CORNERS, atol=6)


def test_side_at_splits_at_the_net():
    table = TableGeometry.from_corners(CORNERS, (W, H))
    assert table.side_at((200, 230)) is Side.LEFT
    assert table.side_at((450, 230)) is Side.RIGHT
    assert table.side_at((320, 50)) is None
    # Just above the surface still counts (ball seen from the side when it touches).
    assert table.side_at((200, 196)) is Side.LEFT


def test_geometry_round_trips_and_rescales(tmp_path):
    table = TableGeometry.from_corners(CORNERS, (W, H))
    table.save(tmp_path / "t.json")
    loaded = TableGeometry.load(tmp_path / "t.json").scaled_to((W * 2, H * 2))
    np.testing.assert_allclose(loaded.corners, table.corners * 2)


def parabola_track(x0, x1, y_table, t0, steps, fps=30.0, apex=60.0):
    """Ball flying from x0 to x1, touching y_table at both ends."""
    out = []
    for i in range(steps + 1):
        s = i / steps
        out.append(BallObservation(t0 + i / fps, x0 + (x1 - x0) * s, y_table - apex * 4 * s * (1 - s), 3, 1))
    return out


def test_bounce_detector_finds_alternating_bounces():
    table = TableGeometry.from_corners(CORNERS, (W, H))
    det = BounceDetector(table)
    track = parabola_track(150, 470, 228, 0.0, 12)[:-1] + parabola_track(470, 180, 228, 0.4, 12)
    bounces = [b for b in (det.update(o) for o in track) if b]
    assert [b.side for b in bounces] == [Side.RIGHT]
    assert abs(bounces[0].t - 0.4) < 0.05


def test_bounce_off_the_table_is_ignored():
    table = TableGeometry.from_corners(CORNERS, (W, H))
    det = BounceDetector(table)
    # Same arc but 80 px below the table: floor bounce.
    track = parabola_track(150, 470, 330, 0.0, 12)[:-1] + parabola_track(470, 180, 330, 0.4, 12)
    assert not any(det.update(o) for o in track)


def test_ball_detector_follows_moving_orange_ball():
    table = TableGeometry.from_corners(CORNERS, (W, H))
    det = BallDetector(table, colour="orange")
    base = scene()
    found = []
    for i in range(10):
        frame = base.copy()
        x, y = 150 + 30 * i, 150
        cv2.circle(frame, (x, y), 4, (0, 140, 255), -1)
        obs = det.detect(frame, i / 30)
        if obs:
            found.append((obs.x, obs.y, x, y))
    assert len(found) >= 7
    for ox, oy, x, y in found:
        assert abs(ox - x) < 3 and abs(oy - y) < 3
