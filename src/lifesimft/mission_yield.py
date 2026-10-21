"""Evaluate mission detection yields for fixed observing schedules."""

from dataclasses import dataclass
from math import isfinite, isnan
from typing import Iterable, Mapping


@dataclass(frozen=True)
class PlanetDetectionTime:
    universe_id: int
    star_id: int
    detection_time_s: float


@dataclass(frozen=True)
class MissionYield:
    universe_ids: tuple[int, ...]
    detections_per_universe: tuple[int, ...]
    expected_detections: float
    probability_at_least_one: float


def evaluate_fixed_allocation(
    records: Iterable[PlanetDetectionTime],
    observing_time_s: Mapping[int, float],
    universe_ids: Iterable[int],
) -> MissionYield:
    """Count planets detectable with a fixed time allocation per star.

    Each target receives the same allocated time in every universe.
    All planets around a star share its observation time.
    """

    ids = tuple(universe_ids)

    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Universe IDs must be nonempty and unique")

    for time_s in observing_time_s.values():
        if not isfinite(time_s) or time_s < 0:
            raise ValueError("Observing times must be finite and nonnegative")

    counts = {universe_id: 0 for universe_id in ids}

    for record in records:
        if record.universe_id not in counts:
            raise ValueError("Unknown universe ID")

        if isnan(record.detection_time_s) or record.detection_time_s < 0:
            raise ValueError("Invalid planet detection time")

        allocated_time = observing_time_s.get(record.star_id, 0.0)

        if allocated_time > 0 and record.detection_time_s <= allocated_time:
            counts[record.universe_id] += 1

    detections = tuple(counts[universe_id] for universe_id in ids)

    n_universes = len(ids)

    return MissionYield(
        universe_ids=ids,
        detections_per_universe=detections,
        expected_detections=sum(detections) / n_universes,
        probability_at_least_one=sum(n >= 1 for n in detections) / n_universes,
    )
