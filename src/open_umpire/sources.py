"""Video and audio inputs: USB cameras, files, and phones streaming over the network.

Video `spec` examples:
    0, /dev/video0                 USB camera (V4L2)
    match.mp4                      file, replayed at its own frame rate timestamps
    http://192.168.1.50:8080/video Android "IP Webcam" app (MJPEG)
    rtsp://...                     anything FFmpeg can open

Audio `spec` examples:
    http://192.168.1.50:8080/audio.wav  IP Webcam microphone (streamed WAV)
    rally.wav                           file
    alsa:plughw:1,0                     ALSA capture device via `arecord` (USB mic, sound card)
    ffmpeg:<input>                      anything FFmpeg can open, e.g. ffmpeg:match.mp4
"""

from __future__ import annotations

import shutil
import struct
import subprocess
import threading
import time
import urllib.request
import wave
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def _is_live(spec: str) -> bool:
    return spec.isdigit() or spec.startswith(("/dev/", "http://", "https://", "rtsp://"))


class VideoSource:
    """Yields (timestamp, frame). Live sources drop frames instead of lagging behind."""

    def __init__(self, spec: str, width: int | None = None, height: int | None = None, fps: int | None = None):
        self.spec = spec
        self.live = _is_live(spec)
        target = int(spec) if spec.isdigit() else spec
        if isinstance(target, int) or spec.startswith("/dev/"):
            self.cap = cv2.VideoCapture(target, cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        else:
            self.cap = cv2.VideoCapture(target)
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open video source {spec!r}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._latest: tuple[float, np.ndarray] | None = None
        self._new = threading.Condition()
        self._stopped = False
        self._index = 0
        self._thread: threading.Thread | None = None
        if self.live:
            self._thread = threading.Thread(target=self._reader, daemon=True)
            self._thread.start()

    def _reader(self) -> None:
        while not self._stopped:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._new:
                self._latest = (time.monotonic(), frame)
                self._new.notify_all()

    def frames(self) -> Iterator[tuple[float, np.ndarray]]:
        if not self.live:
            while not self._stopped:
                ok, frame = self.cap.read()
                if not ok:
                    return
                self._index += 1
                yield self._index / self.fps, frame
            return
        last_t = None
        while not self._stopped:
            with self._new:
                self._new.wait_for(lambda: self._stopped or (self._latest and self._latest[0] != last_t), timeout=2)
                if self._latest is None or self._latest[0] == last_t:
                    continue
                last_t, frame = self._latest
            yield last_t, frame

    def close(self) -> None:
        self._stopped = True
        with self._new:
            self._new.notify_all()
        # Releasing while the reader is inside cap.read() frees FFmpeg state under it (segfault).
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                return
        self.cap.release()


@dataclass
class AudioFormat:
    rate: int
    channels: int


class AudioSource:
    """Yields (timestamp of first sample, mono float32 block in [-1, 1])."""

    def __init__(self, spec: str, rate: int = 44100, block: float = 0.02):
        self.spec = spec
        self.block = block
        self.live = not (Path(spec).suffix.lower() == ".wav" and Path(spec).exists())
        self._proc: subprocess.Popen | None = None
        if spec.startswith(("http://", "https://")):
            self._stream = urllib.request.urlopen(spec, timeout=10)
            self.format = _read_wav_header(self._stream)
        elif spec.startswith("alsa:"):
            self.format = AudioFormat(rate, 1)
            self._spawn(["arecord", "-q", "-D", spec[5:], "-f", "S16_LE", "-r", str(rate), "-c", "1", "-t", "raw"])
        elif spec.startswith("ffmpeg:"):
            self.format = AudioFormat(rate, 1)
            src = spec[7:]
            self.live = _is_live(src)
            self._spawn(
                ["ffmpeg", "-loglevel", "error", *([] if self.live else ["-re"]), "-i", src,
                 "-vn", "-ac", "1", "-ar", str(rate), "-f", "s16le", "-"]
            )
        else:
            w = wave.open(spec, "rb")
            if w.getsampwidth() != 2:
                raise ValueError("only 16-bit WAV files are supported")
            self.format = AudioFormat(w.getframerate(), w.getnchannels())
            self._stream = w
        self._stopped = False

    def _spawn(self, cmd: list[str]) -> None:
        if shutil.which(cmd[0]) is None:
            raise RuntimeError(f"{cmd[0]} is not installed")
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        self._stream = self._proc.stdout

    def _read(self, nbytes: int) -> bytes:
        if isinstance(self._stream, wave.Wave_read):
            return self._stream.readframes(nbytes // (2 * self.format.channels))
        data = b""
        while len(data) < nbytes:
            chunk = self._stream.read(nbytes - len(data))
            if not chunk:
                break
            data += chunk
        return data

    def blocks(self) -> Iterator[tuple[float, np.ndarray]]:
        fmt = self.format
        n = int(fmt.rate * self.block)
        nbytes = n * 2 * fmt.channels
        sample = 0
        while not self._stopped:
            data = self._read(nbytes)
            if len(data) < 2 * fmt.channels:
                return
            pcm = np.frombuffer(data[: len(data) // (2 * fmt.channels) * 2 * fmt.channels], "<i2")
            pcm = pcm.reshape(-1, fmt.channels).mean(axis=1).astype(np.float32) / 32768
            # Live: the block ends now, so it started one block ago. File: its position.
            t = time.monotonic() - len(pcm) / fmt.rate if self.live else sample / fmt.rate
            sample += len(pcm)
            yield t, pcm

    def close(self) -> None:
        self._stopped = True
        if self._proc is not None:
            self._proc.kill()


def _read_wav_header(stream) -> AudioFormat:
    """Parse a (possibly endless) WAV stream header and leave the stream at the sample data."""
    riff = stream.read(12)
    if riff[:4] != b"RIFF" or riff[8:12] != b"WAVE":
        raise ValueError("audio stream is not WAV")
    fmt = None
    while True:
        chunk_id, size = struct.unpack("<4sI", stream.read(8))
        if chunk_id == b"fmt ":
            body = stream.read(size)
            _, channels, rate, _, _, bits = struct.unpack("<HHIIHH", body[:16])
            if bits != 16:
                raise ValueError("only 16-bit audio is supported")
            fmt = AudioFormat(rate, channels)
        elif chunk_id == b"data":
            if fmt is None:
                raise ValueError("WAV data before format")
            return fmt
        else:
            stream.read(size)
