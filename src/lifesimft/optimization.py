"""Greedy observing-time optimization for mission-yield objectives."""

from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Literal

from lifesimft.mission_yield import (
    MissionYield,
    PlanetDetectionTime,
    evaluate_fixed_allocation,
)


@dataclass(frozen=True)
class OptimizationResult:
    allocation_s: dict[int, float]
    used_time_s: float
    mission_yield: MissionYield


def greedy_optimize(
    records: Iterable[PlanetDetectionTime],
    star_ids: Iterable[int],
    universe_ids: Iterable[int],
    total_time_s: float,
    objective: Literal["E", "P"],
) -> OptimizationResult:
    """Greedy optimization of E[N_det] or P(N_det >= 1).

    Candidate observing times correspond to planet detection
    thresholds. This is not guaranteed to find the global optimum.
    """

    if objective not in ("E", "P"):
        raise ValueError("objective must be 'E' or 'P'")

    if not isfinite(total_time_s) or total_time_s < 0:
        raise ValueError("total_time_s must be finite and nonnegative")

    records = tuple(records)
    stars = tuple(star_ids)
    universes = tuple(universe_ids)

    if not stars or len(set(stars)) != len(stars):
        raise ValueError("Star IDs must be nonempty and unique")

    allocation = {star_id: 0.0 for star_id in stars}

    current = evaluate_fixed_allocation(
        records,
        allocation,
        universes,
    )

    candidate_times = {
        star_id: sorted(
            {
                record.detection_time_s
                for record in records
                if record.star_id == star_id
                and isfinite(record.detection_time_s)
                and 0 < record.detection_time_s <= total_time_s
            }
        )
        for star_id in stars
    }

    def objective_score(result: MissionYield) -> int:
        counts = result.detections_per_universe

        if objective == "E":
            return sum(counts)

        return sum(n >= 1 for n in counts)

    while True:
        remaining = total_time_s - sum(allocation.values())

        best_key = None
        best_star = None
        best_time = None
        best_result = None

        current_score = objective_score(current)

        for star_id in sorted(stars):
            for new_time in candidate_times[star_id]:
                delta_time = new_time - allocation[star_id]

                if delta_time <= 0 or delta_time > remaining:
                    continue

                trial_allocation = allocation.copy()
                trial_allocation[star_id] = new_time

                trial_result = evaluate_fixed_allocation(
                    records,
                    trial_allocation,
                    universes,
                )

                gain = objective_score(trial_result) - current_score

                if objective == "P" and gain == 0:
                    # Secondary objective: improve expected yield
                    # without sacrificing detection probability.
                    e_gain = sum(trial_result.detections_per_universe) - sum(
                        current.detections_per_universe
                    )

                    if e_gain <= 0:
                        continue

                    key = (0, e_gain / delta_time, e_gain, -delta_time)

                else:
                    if gain <= 0:
                        continue

                    # Primary-objective improvements always take priority.
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

    return OptimizationResult(
        allocation_s=allocation,
        used_time_s=sum(allocation.values()),
        mission_yield=current,
    )
