"""Tests for Robustness-Lib noise stability."""

import numpy as np
import pytest

from robustness_lib import add_noise, quantize, stability_check


def test_add_noise_zero_std_is_identity():
    a = np.ones((4, 4))
    assert np.array_equal(add_noise(a, std=0.0), a)


def test_add_noise_scales_with_std():
    rng = np.random.default_rng(0)
    a = np.zeros((200, 200))
    noisy = add_noise(a, std=0.1, rng=rng)
    assert noisy.std() == pytest.approx(0.1, rel=0.2)


def test_add_noise_validates():
    with pytest.raises(ValueError):
        add_noise(np.ones((4, 4)), std=-1.0)


def test_quantize_tolerates_noise():
    rng = np.random.default_rng(1)
    a = np.ones((32, 32))
    noisy = add_noise(a, std=0.05, rng=rng)
    q = quantize(noisy, bits=8)
    assert np.linalg.norm(a - q) / np.linalg.norm(a) < 0.2


def test_stability_check_shape_and_sorting():
    rng = np.random.default_rng(2)
    t = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    a = np.tile(np.sin(t), (32, 1))
    rows = stability_check(a, noise_std=0.05, repeats=2, rng=rng)
    assert len(rows) == 4
    errs = [r["mean_relative_error"] for r in rows]
    assert errs == sorted(errs)
    assert all(isinstance(r["stable"], bool) for r in rows)
