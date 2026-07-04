"""Tests for the Ackley function and bounds helpers."""

import numpy as np
import pytest

from src.ackley.function import (
    ackley_value,
    clip_to_bounds,
    default_bounds,
    is_inside_bounds,
    sample_uniform,
)


def test_ackley_at_origin_is_zero():
    assert ackley_value([0.0] * 10) == pytest.approx(0.0, abs=1e-9)


def test_ackley_nonzero_point_is_positive():
    assert ackley_value([1.0] * 10) > 0.0
    assert ackley_value([-5.0, 3.0, 0.5] + [0.0] * 7) > 0.0


def test_ackley_accepts_tuple_and_numpy_array():
    as_list = ackley_value([1.0, 2.0])
    assert ackley_value((1.0, 2.0)) == pytest.approx(as_list)
    assert ackley_value(np.array([1.0, 2.0])) == pytest.approx(as_list)
    assert isinstance(as_list, float)


def test_default_bounds():
    bounds = default_bounds(10)
    assert len(bounds) == 10
    assert all(pair == (-32.768, 32.768) for pair in bounds)


def test_is_inside_bounds():
    bounds = default_bounds(3)
    assert is_inside_bounds([0.0, 10.0, -32.768], bounds)
    assert not is_inside_bounds([0.0, 40.0, 0.0], bounds)
    assert not is_inside_bounds([-33.0, 0.0, 0.0], bounds)


def test_clip_to_bounds():
    bounds = default_bounds(3)
    clipped = clip_to_bounds([100.0, -100.0, 5.0], bounds)
    assert clipped == [32.768, -32.768, 5.0]
    assert is_inside_bounds(clipped, bounds)


def test_sample_uniform_inside_bounds():
    rng = np.random.default_rng(7)
    bounds = default_bounds(10)
    for _ in range(20):
        point = sample_uniform(rng, bounds)
        assert len(point) == 10
        assert is_inside_bounds(point, bounds)
