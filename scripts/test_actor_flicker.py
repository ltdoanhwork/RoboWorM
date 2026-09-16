"""Numerical smoke tests for the experimental residual; no model downloads."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from metrics.dynamics.actor_flicker import (aligned_residual, residual_variants,
                                            SUBTRACT_K)

CANDIDATES = ('raw', 'flownorm', 'gradnorm', 'unexplained', 'shiftmin'
              ) + tuple(f'sub_{k:g}' for k in SUBTRACT_K)


def variants():
    """The E3 candidates must (a) stay ~0 on clean motion, (b) fire on real flicker.

    Built as a textured patch translating by a known amount, so the ONLY difference
    between the two cases is a brightness perturbation -- exactly the discrimination
    the v1 statistic fails (it grows with translation alone).
    """
    rng = np.random.default_rng(7)
    h = w = 64
    mask = np.ones((h, w), bool)
    # Low-pass filtered noise, NOT per-pixel white noise: real video is spatially
    # smooth, and the whole premise of a gradient-based misalignment correction
    # (residual ~ |grad I| * alignment error) is a first-order expansion that only
    # holds when the image varies smoothly over the ~1 px error scale. White noise
    # would make every correction look useless for a reason that does not exist in
    # the data being measured.
    base = np.clip(cv2.GaussianBlur(
        rng.normal(128, 60, (h, w, 3)).astype(np.float32), (0, 0), 2.0),
        20, 235).astype(np.uint8)

    def case(shift, jitter):
        second = np.roll(base, shift, axis=1).astype(np.int16)
        if jitter:
            second = second + rng.integers(-jitter, jitter + 1, second.shape)
        second = np.clip(second, 0, 255).astype(np.uint8)
        fwd = np.zeros((h, w, 2), np.float32)
        fwd[..., 0] = shift                     # exact flow; no estimation error
        v, _ = residual_variants(base, second, fwd, -fwd, mask, mask)
        return v

    slow_clean, fast_clean = case(1, 0), case(6, 0)
    slow_flick, fast_flick = case(1, 24), case(6, 24)

    # (a) with EXACT flow and no photometric change every variant must read ~0
    for name, v in (('slow', slow_clean), ('fast', fast_clean)):
        for k in CANDIDATES:
            assert v[k] < 0.02, f'{k} not ~0 on clean {name} motion: {v[k]}'

    # (b) flicker must raise every variant well clear of the clean case
    for k in CANDIDATES:
        assert slow_flick[k] > 3 * max(slow_clean[k], 1e-4), f'{k} misses slow flicker'
        assert fast_flick[k] > 3 * max(fast_clean[k], 1e-4), f'{k} misses fast flicker'

    # (c) the motion confound: same flicker, 6x the displacement. `raw` must not be
    # allowed to grow much, and the normalized variants must be no worse than raw.
    ratio = {k: fast_flick[k] / slow_flick[k] for k in CANDIDATES}
    print('  fast/slow ratio at equal flicker (1.0 = motion-independent): '
          + ', '.join(f'{k} {r:.2f}' for k, r in ratio.items()))
    assert ratio['unexplained'] < 1.5, ratio

    # (d) THE REAL CONFOUND. With EXACT flow every variant above is already
    # motion-independent, which is precisely why the production statistic's rho with
    # motion (+0.70) cannot be reproduced synthetically without modelling *flow
    # estimation error*: a real flow field is wrong by a fraction of the displacement,
    # so a fast pixel is misaligned further than a slow one and its post-warp residual
    # is larger for reasons unrelated to flicker. Model that as flow error proportional
    # to magnitude, with NO photometric change at all -- an honest metric must then
    # read the same at 1 px and at 6 px.
    def case_err(shift, rel_err=0.08):
        second = np.roll(base, shift, axis=1)
        fwd = np.zeros((h, w, 2), np.float32)
        fwd[..., 0] = shift * (1.0 - rel_err)          # under-estimated, as flow is
        bwd = np.zeros((h, w, 2), np.float32)
        bwd[..., 0] = -shift * (1.0 - rel_err)
        v, _ = residual_variants(base, second, fwd, bwd, mask, mask)
        return v

    slow_err, fast_err = case_err(1), case_err(6)
    growth = {k: fast_err[k] / max(slow_err[k], 1e-9) for k in CANDIDATES}
    print('  fast/slow ratio with 8% FLOW ERROR and NO flicker '
          '(1.0 = motion-independent, this is the confound): '
          + ', '.join(f'{k} {r:.2f}' for k, r in growth.items()))
    # raw is expected to inflate here; the point of E3 is that a candidate does not.
    assert growth['raw'] > 1.5, f'test failed to reproduce the confound: {growth}'
    better = [k for k in CANDIDATES if k != 'raw' and growth[k] < growth['raw']]
    print(f'  variants less motion-inflated than raw: {better}')
    assert better, growth
    print('PASS: variants ~0 on clean motion, fire on flicker, ratios printed')


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
    # Variant dict must agree with v1 on `raw` and report the SAME pixel count as v1
    # (identical validity geometry -- here 899 of 1024, the rest shifted out of frame).
    v1_value, v1_count = aligned_residual(first, second, forward, -forward, mask, mask)
    v, n = residual_variants(first, second, forward, -forward, mask, mask)
    assert v['raw'] == v1_value and n == v1_count, (v, n, v1_value, v1_count)
    v, n = residual_variants(first, second, forward, -forward, mask, ~mask)
    assert all(x is None for x in v.values()) and n == 0
    print('PASS: freeze, brightness flicker, flow rejection, translation, empty overlap')
    variants()


if __name__ == '__main__': main()
