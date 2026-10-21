"""Greedy observing-time optimization for mission-yield objectives."""

from bisect import bisect_right
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
    """Greedily optimize E[N_det] or P(N_det >= 1).

    Score only the newly detectable planets for a candidate time increase,
    instead of evaluating the entire population for every trial allocation.
    One common observing-time allocation applies to all simulated universes.

    The candidate times, tie-breaking, and P objective's E-yield fallback
    match the original greedy implementation. Global optimality is not guaranteed.
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

    # Retain the input validation and zero-allocation semantics of the
    # original mission-yield evaluator, including records for other stars.
    evaluate_fixed_allocation(records, allocation, universes)

    # Each star has a sorted event stream: (detection threshold, universe).
    # Keep threshold==0 events: these become detectable once a star receives
    # any strictly positive allocation. Infinite times cannot be candidates.
    events: dict[int, list[tuple[float, int]]] = {star_id: [] for star_id in stars}
    for record in records:
        if (
            record.star_id in events
            and isfinite(record.detection_time_s)
            and record.detection_time_s <= total_time_s
        ):
            events[record.star_id].append((record.detection_time_s, record.universe_id))
    for star_events in events.values():
        star_events.sort()

    # Tracks planets already detected at the current allocation for each star.
    processed = {star_id: 0 for star_id in stars}
    # Current planet counts, indexed by universe ID (no dense Nstars x Nuniv grid).
    counts = {universe_id: 0 for universe_id in universes}

    while True:
        remaining = total_time_s - sum(allocation.values())
        best_key = None
        best_star = None
        best_time = None

        for star_id in sorted(stars):
            star_events = events[star_id]
            star_time = allocation[star_id]
            i = processed[star_id]
            e_gain = 0
            p_gain = 0
            newly_covered: set[int] = set()

            # Scan each candidate threshold once per star per greedy step.
            # Only events after the star's existing allocation contribute.
            while i < len(star_events):
                new_time = star_events[i][0]
                if new_time > 0 and new_time - star_time > remaining:
                    break

                # Aggregate simultaneous detections before evaluating the
                # candidate. This preserves ties at duplicate thresholds.
                while i < len(star_events) and star_events[i][0] == new_time:
                    universe_id = star_events[i][1]
                    e_gain += 1
                    if counts[universe_id] == 0 and universe_id not in newly_covered:
                        newly_covered.add(universe_id)
                        p_gain += 1
                    i += 1

                # A zero detection threshold is *not* an allowed allocation,
                # but zero-time planets must count at any positive candidate.
                if new_time <= 0:
                    continue

                delta_time = new_time - star_time
                if delta_time <= 0:
                    continue

                gain = e_gain if objective == "E" else p_gain
                if objective == "P" and gain == 0:
                    if e_gain <= 0:
                        continue
                    key = (0, e_gain / delta_time, e_gain, -delta_time)
                else:
                    if gain <= 0:
                        continue
                    key = (1, gain / delta_time, gain, -delta_time)

                # Strict '>' preserves original deterministic tie-breaking:
                # smaller star ID, then smaller candidate threshold.
                if best_key is None or key > best_key:
                    best_key = key
                    best_star = star_id
                    best_time = new_time

        if best_star is None:
            break

        allocation[best_star] = best_time
        star_events = events[best_star]
        first = processed[best_star]
        last = bisect_right(star_events, (best_time, float("inf")))
        for _, universe_id in star_events[first:last]:
            counts[universe_id] += 1
        processed[best_star] = last

    # Keep the public MissionYield construction in one place and perform an
    # independent final evaluation as a correctness safeguard.
    final_yield = evaluate_fixed_allocation(records, allocation, universes)
    assert final_yield.detections_per_universe == tuple(
        counts[universe_id] for universe_id in universes
    ), "Incremental detection counts disagree with fixed-allocation evaluator"

    return OptimizationResult(
        allocation_s=allocation,
        used_time_s=sum(allocation.values()),
        mission_yield=final_yield,
    )
