"""Compare original and incremental greedy optimizers on the 100-universe P-Pop data.

Place alongside benchmark_detection_workers.py and run on macOS:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      python examples/compare_optimizer_100.py --workers 18

The two strategies must yield exactly the same allocations and mission results.
"""

from __future__ import annotations

import argparse
from math import isfinite
from pathlib import Path
from time import perf_counter

from benchmark_detection_workers import compute

from lifesimft.constants import R_EARTH_M
from lifesimft.mission_yield import PlanetDetectionTime, evaluate_fixed_allocation
from lifesimft.optimization import greedy_optimize
from lifesimft.population import load_ppop


def original_greedy(records, star_ids, universe_ids, total_time_s, objective):
    """Unmodified original greedy scoring loop, for verification only."""
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
                and isfinite(record.detection_time_s)
                and 0 < record.detection_time_s <= total_time_s
            }
        )
        for star_id in stars
    }

    def objective_score(result):
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
                    records, trial_allocation, universes
                )
                gain = objective_score(trial_result) - current_score
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path.home() / "Developer/P-pop/YieldPopulation100.txt",
    )
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--chunksize", type=int, default=8)
    args = parser.parse_args()

    t0 = perf_counter()
    population = load_ppop(args.catalog)
    candidates = [
        e
        for e in population
        if e.star_id in range(10)
        and 0.5 <= e.planet.radius_m / R_EARTH_M <= 1.5
        and 200 <= e.planet.temperature_k <= 320
    ]
    print(f"Eligible planets: {len(candidates)}", flush=True)
    if not candidates:
        raise RuntimeError("No eligible planets")

    start = perf_counter()
    detection_times = compute(candidates, args.workers, args.chunksize)
    records = [
        PlanetDetectionTime(e.universe_id, e.star_id, time_s)
        for e, time_s in zip(candidates, detection_times)
    ]
    print(
        f"Planet detection times ({args.workers} workers): "
        f"{perf_counter() - start:.2f} s",
        flush=True,
    )

    kwargs = dict(
        records=records,
        star_ids=list(range(10)),
        universe_ids=list(range(100)),
        total_time_s=50 * 3600.0,
    )
    print("\nObjective   Original s  Improved s  Speedup   Identical")
    for objective in ("E", "P"):
        start = perf_counter()
        old_allocation, old_time, old_yield = original_greedy(
            **kwargs, objective=objective
        )
        t_old = perf_counter() - start
        start = perf_counter()
        new = greedy_optimize(**kwargs, objective=objective)
        t_new = perf_counter() - start
        identical = (
            new.allocation_s == old_allocation
            and new.used_time_s == old_time
            and new.mission_yield == old_yield
        )
        print(
            f"{objective:>9} {t_old:11.3f} {t_new:11.3f} "
            f"{t_old / t_new:8.2f}x {'YES' if identical else 'NO'}",
            flush=True,
        )
        if not identical:
            print("Original:", old_allocation, old_yield)
            print("New:", new.allocation_s, new.mission_yield)
            raise AssertionError(f"{objective}-optimizer results differ")
        print(
            f"  E[N]={new.mission_yield.expected_detections:.4f}, "
            f"P(N>=1)={new.mission_yield.probability_at_least_one:.4f}, "
            f"used={new.used_time_s / 3600:.2f} h",
            flush=True,
        )

    print(f"\nTotal wall time: {perf_counter() - t0:.2f} s", flush=True)
    print("PASS: Both optimizers match exactly on the 100-universe population.")


if __name__ == "__main__":
    main()
