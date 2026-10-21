"""Tests for greedy mission-yield optimization."""

import pytest

from lifesimft.mission_yield import PlanetDetectionTime
from lifesimft.optimization import greedy_optimize


def test_e_and_p_choose_different_allocations():
    """E favors multiplicity; P favors covering more universes."""

    records = [
        # Five planets in universe 0, around star 0.
        *[PlanetDetectionTime(0, 0, 3600.0) for _ in range(5)],
        # One planet each in universes 1 and 2, around star 1.
        PlanetDetectionTime(1, 1, 1800.0),
        PlanetDetectionTime(2, 1, 1800.0),
    ]

    settings = dict(
        records=records,
        star_ids=[0, 1],
        universe_ids=range(3),
        total_time_s=3600.0,
    )

    e_result = greedy_optimize(**settings, objective="E")
    p_result = greedy_optimize(**settings, objective="P")

    assert e_result.allocation_s[0] == pytest.approx(3600.0)
    assert p_result.allocation_s[1] == pytest.approx(1800.0)

    assert e_result.mission_yield.expected_detections == pytest.approx(5 / 3)
    assert p_result.mission_yield.probability_at_least_one == pytest.approx(2 / 3)

    assert e_result.used_time_s <= 3600.0
    assert p_result.used_time_s <= 3600.0


def test_undetectable_planets():
    records = [PlanetDetectionTime(0, 0, float("inf"))]

    result = greedy_optimize(
        records=records,
        star_ids=[0],
        universe_ids=[0, 1],
        total_time_s=3600.0,
        objective="E",
    )

    assert result.used_time_s == 0.0
    assert result.mission_yield.expected_detections == 0.0


def test_invalid_optimizer_inputs():
    settings = dict(
        records=[],
        star_ids=[0],
        universe_ids=[0],
    )

    with pytest.raises(ValueError):
        greedy_optimize(**settings, total_time_s=-1, objective="E")

    with pytest.raises(ValueError):
        greedy_optimize(**settings, total_time_s=3600, objective="invalid")


def test_p_optimizer_uses_remaining_time_for_expected_yield():
    """Once P is maximized, additional time can increase E."""

    records = [
        PlanetDetectionTime(0, 0, 3600.0),
        PlanetDetectionTime(0, 0, 7200.0),
    ]

    result = greedy_optimize(
        records=records,
        star_ids=[0],
        universe_ids=[0],
        total_time_s=7200.0,
        objective="P",
    )

    assert result.used_time_s == pytest.approx(7200.0)
    assert result.mission_yield.probability_at_least_one == 1.0
    assert result.mission_yield.expected_detections == 2.0
