"""First fixed-allocation mission-yield experiment."""

from pathlib import Path
from time import perf_counter

from lifesimft.constants import R_EARTH_M
from lifesimft.detectability import calculate_detection_time
from lifesimft.mission_yield import (
    PlanetDetectionTime,
    evaluate_fixed_allocation,
)
from lifesimft.optimization import greedy_optimize
from lifesimft.population import load_ppop

N_UNIVERSES = 10
N_TARGETS = 10
TIME_PER_STAR_H_GRID = (1.0, 2.0, 5.0, 10.0, 20.0)
SNR_THRESHOLD = 5.0


def main():
    catalog = Path.home() / "Developer/P-pop/TestPlanetPopulation.txt"

    population = load_ppop(catalog)

    # Fixed target selection, independent of simulated planets.
    target_ids = set(range(N_TARGETS))

    # Preliminary Earth-like population definition.
    candidates = [
        entry
        for entry in population
        if entry.star_id in target_ids
        and 0.5 <= entry.planet.radius_m / R_EARTH_M <= 1.5
        and 200 <= entry.planet.temperature_k <= 320
    ]

    print(f"Universes: {N_UNIVERSES}")
    print(f"Target stars: {N_TARGETS}")
    print(f"Time allocations: {TIME_PER_STAR_H_GRID} h per star")
    print(f"Eligible simulated planets: {len(candidates)}")

    # Calculate planet detection times only once.
    start = perf_counter()

    records = []

    for entry in candidates:
        t_det = calculate_detection_time(
            entry.star,
            entry.planet,
            snr_threshold=SNR_THRESHOLD,
        )

        records.append(
            PlanetDetectionTime(
                universe_id=entry.universe_id,
                star_id=entry.star_id,
                detection_time_s=t_det,
            )
        )

    elapsed = perf_counter() - start

    print(f"\nDetection-time calculations: {elapsed:.2f} s")

    # The same target list and allocation apply in every universe.
    print("\nFIXED-ALLOCATION COMPARISON")
    print(f"{'h/star':>8} {'Total h':>10} {'E[N_det]':>10} {'P(N>=1)':>10}")

    previous_e = 0.0
    previous_p = 0.0

    for time_h in TIME_PER_STAR_H_GRID:
        allocation = {star_id: time_h * 3600.0 for star_id in target_ids}

        result = evaluate_fixed_allocation(
            records,
            allocation,
            universe_ids=range(N_UNIVERSES),
        )

        print(
            f"{time_h:8.1f} "
            f"{N_TARGETS * time_h:10.1f} "
            f"{result.expected_detections:10.4f} "
            f"{result.probability_at_least_one:10.4f}"
        )

        # More observing time cannot reduce deterministic yield.
        assert result.expected_detections >= previous_e
        assert result.probability_at_least_one >= previous_p

        previous_e = result.expected_detections
        previous_p = result.probability_at_least_one

    # -----------------------------------------------------------
    # Compare E- and P-optimized observing strategies.
    # -----------------------------------------------------------

    budget_h = 50.0

    strategies = {}

    for objective in ("E", "P"):
        strategies[objective] = greedy_optimize(
            records=records,
            star_ids=sorted(target_ids),
            universe_ids=range(N_UNIVERSES),
            total_time_s=budget_h * 3600.0,
            objective=objective,
        )

    print("\nOPTIMIZED STRATEGIES")
    print(f"Available observing time: {budget_h:.1f} h")
    print(f"{'Strategy':<12} {'E[N_det]':>10} {'P(N>=1)':>10} {'Used h':>10}")

    for objective, result in strategies.items():
        print(
            f"{objective + '-optimized':<12} "
            f"{result.mission_yield.expected_detections:10.4f} "
            f"{result.mission_yield.probability_at_least_one:10.4f} "
            f"{result.used_time_s / 3600.0:10.2f}"
        )

        print("  Allocation (star ID: hours):")

        for star_id, time_s in sorted(result.allocation_s.items()):
            if time_s > 0:
                print(f"    {star_id}: {time_s / 3600.0:.2f}")

    e_opt = strategies["E"].mission_yield
    p_opt = strategies["P"].mission_yield

    delta_p = p_opt.probability_at_least_one - e_opt.probability_at_least_one

    delta_e = e_opt.expected_detections - p_opt.expected_detections

    print("\nSTRATEGY COMPARISON")
    print(f"Probability advantage of P-optimization: {delta_p:.4f}")
    print(f"Expected-yield advantage of E-optimization: {delta_e:.4f}")


if __name__ == "__main__":
    main()
