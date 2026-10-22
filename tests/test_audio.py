import numpy as np

from open_umpire.audio import OnsetDetector

RATE = 16000


def clicks_signal(times, duration=3.0, noise=0.002, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, noise, int(duration * RATE)).astype(np.float32)
    # Low-frequency hum that must not trigger.
    x += 0.05 * np.sin(2 * np.pi * 50 * np.arange(len(x)) / RATE).astype(np.float32)
    for t in times:
        i = int(t * RATE)
        n = int(0.01 * RATE)
        decay = np.exp(-np.arange(n) / (0.002 * RATE))
        x[i : i + n] += (0.5 * decay * np.sin(2 * np.pi * 3000 * np.arange(n) / RATE)).astype(np.float32)
    return x


def run(x, block=320):
    det = OnsetDetector(RATE)
    found = []
    for start in range(0, len(x), block):
        found += det.process(start / RATE, x[start : start + block])
    return [o.t for o in found]


def test_detects_each_click_once_within_10ms():
    times = [1.2, 1.5, 2.1, 2.6]
    found = run(clicks_signal(times))
    assert len(found) == len(times)
    for expected, got in zip(times, found):
        assert abs(expected - got) < 0.01


def test_hum_and_noise_alone_trigger_nothing():
    assert run(clicks_signal([])) == []


def test_block_size_does_not_change_result():
    x = clicks_signal([1.3, 2.0])
    assert run(x, block=160) == run(x, block=1024)
