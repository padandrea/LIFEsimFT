"""Regress incremental greedy optimization against the original full scans."""

import random

import pytest

from lifesimft.mission_yield import PlanetDetectionTime, evaluate_fixed_allocation
from lifesimft.optimization import greedy_optimize


def original_greedy(records, star_ids, universe_ids, total_time_s, objective):
    """Original brute-force algorithm, retained only as a test oracle."""
    records = tuple(records)
    stars = tuple(star_ids)
    universes = tuple(universe_ids)
    allocation = {star_id: 0.0 for star_id in stars}
    current = evaluate_fixed_allocation(records, allocation, universes)
    candidate_times = {
        star_id: sorted(
            {
                record.detection_time_s
                for record in records
                if record.star_id == star_id
                and record.detection_time_s != float("inf")
                and 0 < record.detection_time_s <= total_time_s
            }
        )
        for star_id in stars
    }

    def score(result):
        counts = result.detections_per_universe
        return sum(counts) if objective == "E" else sum(n >= 1 for n in counts)

    while True:
        remaining = total_time_s - sum(allocation.values())
        best_key = None
        best_star = None
        best_time = None
        best_result = None
        current_score = score(current)
        for star_id in sorted(stars):
            for new_time in candidate_times[star_id]:
                delta_time = new_time - allocation[star_id]
                if delta_time <= 0 or delta_time > remaining:
                    continue
                trial_allocation = allocation.copy()
                trial_allocation[star_id] = new_time
                trial_result = evaluate_fixed_allocation(
                    records, trial_allocation, universes
                )
                gain = score(trial_result) - current_score
                if objective == "P" and gain == 0:
                    e_gain = sum(trial_result.detections_per_universe) - sum(
                        current.detections_per_universe
                    )
                    if e_gain <= 0:
                        continue
                    key = (0, e_gain / delta_time, e_gain, -delta_time)
                else:
                    if gain <= 0:
                        continue
                    key = (1, gain / delta_time, gain, -delta_time)
                if best_key is None or key > best_key:
                    best_key = key
                    best_star = star_id
                    best_time = new_time
                    best_result = trial_result
        if best_star is None:
            break
        allocation[best_star] = best_time
        current = best_result
    return allocation, sum(allocation.values()), current


@pytest.mark.parametrize("objective", ["E", "P"])
@pytest.mark.parametrize("seed", range(25))
def test_matches_original_on_random_small_catalogs(seed, objective):
    rng = random.Random(seed)
    stars = [0, 1, 2, 3]
    universes = list(range(7))
    choices = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, float("inf")]
    records = [
        PlanetDetectionTime(
            universe_id=rng.choice(universes),
            star_id=rng.choice(stars),
            detection_time_s=rng.choice(choices),
        )
        for _ in range(35)
    ]
    # Include planets orbiting a non-target star: original evaluator checks
    # them but they should have no influence on the target optimization.
    records.append(PlanetDetectionTime(0, 99, 0.5))
    settings = dict(
        records=records,
        star_ids=stars,
        universe_ids=universes,
        total_time_s=8.0,
        objective=objective,
    )
    old_allocation, old_time, old_yield = original_greedy(**settings)
    new_result = greedy_optimize(**settings)
    assert new_result.allocation_s == old_allocation
    assert new_result.used_time_s == old_time
    assert new_result.mission_yield == old_yield


@pytest.mark.parametrize("objective", ["E", "P"])
def test_zero_threshold_planets_and_duplicate_events(objective):
    settings = dict(
        records=[
            PlanetDetectionTime(0, 0, 0.0),
            PlanetDetectionTime(0, 0, 0.0),
            PlanetDetectionTime(0, 0, 1.0),
            PlanetDetectionTime(1, 0, 1.0),
            PlanetDetectionTime(1, 1, 1.0),
            PlanetDetectionTime(2, 1, float("inf")),
        ],
        star_ids=[1, 0],
        universe_ids=[2, 0, 1],
        total_time_s=2.0,
        objective=objective,
    )
    old_allocation, old_time, old_yield = original_greedy(**settings)
    new_result = greedy_optimize(**settings)
    assert new_result.allocation_s == old_allocation
    assert new_result.used_time_s == old_time
    assert new_result.mission_yield == old_yield


def test_invalid_planet_time_is_still_rejected():
    records = [PlanetDetectionTime(0, 0, float("nan"))]
    with pytest.raises(ValueError, match="Invalid planet detection time"):
        greedy_optimize(records, [0], [0], 1.0, "E")
