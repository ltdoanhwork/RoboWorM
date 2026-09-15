"""Numerical smoke tests for the experimental residual; no model downloads."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from metrics.dynamics.actor_flicker import aligned_residual


def main():
    a = np.full((32, 32, 3), 100, np.uint8)
    z = np.zeros((32, 32, 2), np.float32)
    mask = np.ones((32, 32), bool)
    value, count = aligned_residual(a, a, z, z, mask, mask)
    assert value == 0 and count > 0
    value, count = aligned_residual(a, a + 30, z, z, mask, mask)
    assert abs(value - 30/255) < 1e-6
    bad = z.copy(); bad[..., 0] = 20
    value, count = aligned_residual(a, a, z, bad, mask, mask)
    assert value is None and count == 0
    rng = np.random.default_rng(42)
    first = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
    second = np.roll(first, 2, axis=1)
    forward = z.copy(); forward[..., 0] = 2
    value, count = aligned_residual(first, second, forward, -forward, mask, mask)
    assert value == 0 and count > 0
    value, count = aligned_residual(first, second, forward, -forward, mask, ~mask)
    assert value is None and count == 0
    print('PASS: freeze, brightness flicker, flow rejection, translation, empty overlap')


if __name__ == '__main__': main()
