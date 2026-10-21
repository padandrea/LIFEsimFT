"""Tests for fixed-allocation mission yield."""

import pytest

from lifesimft.mission_yield import (
    PlanetDetectionTime,
    evaluate_fixed_allocation,
)


def test_fixed_allocation():
    """Count multiple planets per star and preserve empty universes."""

    records = [
        PlanetDetectionTime(0, 0, 3600.0),
        PlanetDetectionTime(0, 0, 7200.0),
        PlanetDetectionTime(1, 0, 25200.0),
        PlanetDetectionTime(1, 1, 3600.0),
        PlanetDetectionTime(1, 2, 3600.0),
    ]

    allocation = {
        0: 18000.0,
        1: 3600.0,
    }

    result = evaluate_fixed_allocation(
        records,
        allocation,
        universe_ids=range(3),
    )

    assert result.detections_per_universe == (2, 1, 0)
    assert result.expected_detections == pytest.approx(1.0)
    assert result.probability_at_least_one == pytest.approx(2 / 3)


def test_empty_universes():
    result = evaluate_fixed_allocation(
        [],
        {0: 3600.0},
        universe_ids=range(3),
    )

    assert result.detections_per_universe == (0, 0, 0)
    assert result.expected_detections == 0.0
    assert result.probability_at_least_one == 0.0


def test_zero_observing_time():
    result = evaluate_fixed_allocation(
        [PlanetDetectionTime(0, 0, 0.0)],
        {0: 0.0},
        universe_ids=[0],
    )

    assert result.detections_per_universe == (0,)
