"""Ties camera, microphone and rules together, and draws the overlay."""

from __future__ import annotations

import json
import queue
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .audio import Onset
from .ball import BallDetector, Bounce, BounceDetector
from .rules import Match, PointEvent, Referee
from .table import TableGeometry


@dataclass
class UmpireConfig:
    ball_colour: str | None = "orange"
    # "vision": bounces from the ball trajectory. "audio": bounces from sound, placed by the camera.
    bounce_source: str = "vision"
    # Seconds to add to audio timestamps to line them up with video (positive if audio arrives early).
    audio_offset: float = 0.0
    log_path: Path | None = None


class Umpire:
    def __init__(self, table: TableGeometry, match: Match, config: UmpireConfig | None = None):
        self.config = config or UmpireConfig()
        self.table = table
        self.match = match
        self.referee = Referee(match)
        self.ball = BallDetector(table, colour=self.config.ball_colour)
        self.bounces = BounceDetector(table)
        self.trail: deque[tuple[int, int]] = deque(maxlen=12)
        self.recent_bounces: deque[Bounce] = deque(maxlen=6)
        self.log: deque[str] = deque(maxlen=50)
        self.onsets: queue.Queue[Onset] = queue.Queue()
        self._pending_onsets: list[Onset] = []
        self._fps_times: deque[float] = deque(maxlen=30)
        self.now = 0.0

    # -- inputs ---------------------------------------------------------------

    def process_frame(self, t: float, frame: np.ndarray) -> None:
        self.now = t
        self._fps_times.append(time.monotonic())
        obs = self.ball.detect(frame, t)
        if obs is not None:
            self.trail.append((int(obs.x), int(obs.y)))
        vision_bounce = self.bounces.update(obs)
        if self.config.bounce_source == "vision" and vision_bounce is not None:
            self._on_bounce(vision_bounce)
        if self.config.bounce_source == "audio":
            self._resolve_onsets(t)
        event = self.referee.tick(t)
        if event:
            self._on_point(event)

    def add_onset(self, onset: Onset) -> None:
        """Thread-safe: called from the audio thread."""
        self.onsets.put(onset)

    # -- manual corrections ---------------------------------------------------

    def command(self, name: str) -> None:
        if name in ("point0", "point1"):
            self.referee.cancel_rally()
            self.match.add_point(int(name[-1]))
            self._note(f"manual point for {self.match.names[int(name[-1])]}")
        elif name == "undo" and self.match.undo():
            self.referee.cancel_rally()
            self._note("undo")
        elif name == "swap":
            self.match.swap_ends()
            self._note("ends swapped")
        elif name == "reset":
            self.match.__post_init__()
            self.referee.cancel_rally()
            self._note("match reset")
        elif name == "let":
            self.referee.cancel_rally()
            self._note("let / rally cancelled")

    # -- internals -------------------------------------------------------------

    def _resolve_onsets(self, t: float) -> None:
        while not self.onsets.empty():
            self._pending_onsets.append(self.onsets.get_nowait())
        keep = []
        for onset in self._pending_onsets:
            ot = onset.t + self.config.audio_offset
            if t - ot < 0.1:  # wait until frames around the sound have been seen
                keep.append(onset)
                continue
            obs = self.bounces.position_at(ot)
            side = self.table.side_at((obs.x, obs.y), margin=self.bounces.margin) if obs else None
            if side is None:
                self._note(f"sound at {ot:.2f}s off table (hit?)")
                continue
            self._on_bounce(Bounce(ot, obs.x, obs.y, side))
        self._pending_onsets = keep

    def _on_bounce(self, bounce: Bounce) -> None:
        self.recent_bounces.append(bounce)
        self._note(f"bounce {bounce.side.value}")
        event = self.referee.bounce(bounce.side, bounce.t)
        if event:
            self._on_point(event)

    def _on_point(self, event: PointEvent) -> None:
        name = self.match.names[event.winner]
        a, b = self.match.scores
        self._note(f"point {name}: {event.reason} ({a}-{b})")
        if self.config.log_path:
            with open(self.config.log_path, "a") as f:
                f.write(json.dumps({"t": event.t, "winner": event.winner, "reason": event.reason,
                                    **self.match.to_dict()}) + "\n")

    def _note(self, text: str) -> None:
        self.log.appendleft(f"{self.now:8.2f}  {text}")

    @property
    def fps(self) -> float:
        if len(self._fps_times) < 2:
            return 0.0
        return (len(self._fps_times) - 1) / max(self._fps_times[-1] - self._fps_times[0], 1e-6)

    def state(self) -> dict:
        return {**self.match.to_dict(), "in_rally": self.referee.in_rally, "fps": round(self.fps, 1),
                "log": list(self.log)[:15]}

    # -- drawing ---------------------------------------------------------------

    def draw(self, frame: np.ndarray) -> np.ndarray:
        out = frame.copy()
        self.table.draw(out)
        for i in range(1, len(self.trail)):
            cv2.line(out, self.trail[i - 1], self.trail[i], (0, 255, 255), 2)
        for b in self.recent_bounces:
            age = self.now - b.t
            if age < 2:
                cv2.circle(out, (int(b.x), int(b.y)), 8, (0, 0, 255), 2)
        m = self.match
        left, right = m.left_player, 1 - m.left_player
        h, w = out.shape[:2]
        scale = max(w / 960, 0.5)
        cv2.rectangle(out, (0, 0), (w, int(44 * scale)), (0, 0, 0), -1)
        for player, x, align in ((left, 10, "l"), (right, w - 10, "r")):
            serve = "*" if m.server == player and not m.finished else " "
            text = f"{serve}{m.names[player]}  {m.games[player]} | {m.scores[player]}"
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
            pos = (x if align == "l" else x - tw, int(32 * scale))
            cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, f"{self.fps:4.1f} fps", (w // 2 - int(40 * scale), int(30 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (180, 180, 180), 1, cv2.LINE_AA)
        if self.log:
            cv2.putText(out, self.log[0].strip(), (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale,
                        (255, 255, 255), 1, cv2.LINE_AA)
        return out
