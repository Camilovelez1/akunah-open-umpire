"""End to end: rendered frames of an orange ball rallying over a synthetic table."""

import cv2
import numpy as np

from open_umpire.audio import Onset
from open_umpire.rules import Match
from open_umpire.table import TableGeometry
from open_umpire.umpire import Umpire, UmpireConfig

from test_vision import CORNERS, H, W, scene

FPS = 60.0
TABLE_Y = 228


def rally(bounce_xs, flight=0.4):
    """(t, x, y) of a ball bouncing at each x in turn, sampled at FPS."""
    pts = []
    for i in range(len(bounce_xs) - 1):
        x0, x1 = bounce_xs[i], bounce_xs[i + 1]
        n = int(flight * FPS)
        for k in range(n):
            s = k / n
            pts.append((x0 + (x1 - x0) * s, TABLE_Y - 70 * 4 * s * (1 - s)))
    pts.append((bounce_xs[-1], TABLE_Y))
    # The ball leaves the table after the last bounce and disappears.
    return [(i / FPS, x, y) for i, (x, y) in enumerate(pts)]


def play(umpire, points, extra_seconds=3.0):
    base = scene()
    end = points[-1][0] + extra_seconds
    i = 0
    t = 0.0
    while t <= end:
        frame = base.copy()
        if i < len(points):
            _, x, y = points[i]
            cv2.circle(frame, (int(x), int(y)), 4, (0, 140, 255), -1)
        umpire.process_frame(t, frame)
        i += 1
        t = i / FPS


def test_vision_rally_awards_point_to_last_hitter():
    table = TableGeometry.from_corners(CORNERS, (W, H))
    match = Match()
    umpire = Umpire(table, match, UmpireConfig(ball_colour="orange"))
    # Toss at x=150, then bounces left, right, left, right; the right player misses and
    # the ball flies off past the end of the table.
    play(umpire, rally([150, 220, 460, 200, 470, 640]))
    assert match.scores == [1, 0], list(umpire.log)


def test_audio_bounces_are_placed_by_the_camera():
    table = TableGeometry.from_corners(CORNERS, (W, H))
    match = Match()
    umpire = Umpire(table, match, UmpireConfig(ball_colour="orange", bounce_source="audio"))
    pts = rally([150, 220, 460])
    # A sound at each bounce, plus one mid-flight (a paddle hit) that must be ignored.
    for t, x, y in pts:
        if y == TABLE_Y and x in (220, 460):
            umpire.add_onset(Onset(t, 30.0))
    umpire.add_onset(Onset(pts[len(pts) // 4][0], 30.0))
    play(umpire, pts)
    assert any("off table" in line for line in umpire.log)
    # Serve left -> right is good, then the right player doesn't return it.
    assert match.scores == [1, 0], list(umpire.log)
