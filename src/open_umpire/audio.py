"""Bounce sounds: short, sharp clicks well above the room's background level.

The audio says *when* the ball touched something; the camera says *where*
(on which half, or off the table for a paddle hit).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Onset:
    t: float
    level: float  # peak energy over background, in dB


class OnsetDetector:
    def __init__(self, rate: int, hop: float = 0.004, threshold_db: float = 12.0, refractory: float = 0.08,
                 min_level: float = 0.01):
        self.rate = rate
        self.hop = max(int(rate * hop), 1)
        self.threshold_db = threshold_db
        self.refractory = refractory
        self.min_level = min_level
        self._background: deque[float] = deque(maxlen=int(1.0 / hop))  # ~1 s of hop energies
        self._carry = np.zeros(0, np.float32)
        self._carry_t = 0.0
        self._prev_sample = 0.0
        self._last_onset = float("-inf")

    def process(self, t: float, block: np.ndarray) -> list[Onset]:
        """Feed a mono block starting at time `t`; returns onsets found in it."""
        if len(self._carry) == 0:
            self._carry_t = t
        # First difference is a cheap high-pass: bounces are clicks, voices and hum are low.
        diff = np.diff(np.concatenate([[self._prev_sample], block])).astype(np.float32)
        self._prev_sample = float(block[-1]) if len(block) else self._prev_sample
        data = np.concatenate([self._carry, diff])
        onsets = []
        n_hops = len(data) // self.hop
        for i in range(n_hops):
            frame = data[i * self.hop : (i + 1) * self.hop]
            rms = float(np.sqrt(np.mean(frame * frame)) + 1e-9)
            ht = self._carry_t + i * self.hop / self.rate
            bg = float(np.median(self._background)) if len(self._background) >= 10 else None
            if bg is not None and rms >= self.min_level and ht - self._last_onset >= self.refractory:
                level = 20 * np.log10(rms / (bg + 1e-9))
                if level >= self.threshold_db:
                    self._last_onset = ht
                    onsets.append(Onset(ht, float(level)))
            self._background.append(rms)
        self._carry = data[n_hops * self.hop :]
        self._carry_t += n_hops * self.hop / self.rate
        return onsets
