"""Compare E/P strategies for LTC4 planets in the conservative HZ."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
from benchmark_detection_workers import compute
from scipy.stats import binomtest

from lifesimft.constants import R_EARTH_M
from lifesimft.habitable_zone import in_conservative_hz
from lifesimft.mission_yield import PlanetDetectionTime, evaluate_fixed_allocation
from lifesimft.optimization import greedy_optimize
from lifesimft.population import load_ppop
from lifesimft.experiment_io import load_or_compute_times, save_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path.home() / "Developer/P-pop/LTC4_BrysonLow_100.txt",
    )
    parser.add_argument("--universes", type=int, default=100)
    parser.add_argument("--train-universes", type=int, default=None)
    parser.add_argument("--budget-hours", type=float, default=50.0)
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--recompute-detection-times", action="store_true")
    args = parser.parse_args()

    population = load_ppop(args.catalog)
    star_ids = sorted({entry.star_id for entry in population})
    star_names = {entry.star_id: entry.star_name for entry in population}

    candidates = [
        entry
        for entry in population
        if 0.5 <= entry.planet.radius_m / R_EARTH_M <= 1.5
        and in_conservative_hz(
            entry.incident_flux_searth,
            entry.star.temperature_k,
        )
    ]

    print(f"Universes: {args.universes}", flush=True)
    print(f"Candidate stars: {len(star_ids)}", flush=True)
    print(f"HZ planets: {len(candidates)}", flush=True)
    print(f"Budget: {args.budget_hours:.1f} h", flush=True)

    assert all(0 <= entry.universe_id < args.universes for entry in population)
    n_train = args.universes if args.train_universes is None else args.train_universes

    if not 1 <= n_train <= args.universes:
        parser.error("--train-universes must be between 1 and --universes")

    train_ids = range(n_train)
    test_ids = range(n_train, args.universes)

    print(f"Training universes: {len(train_ids)}")
    print(f"Evaluation universes: {len(test_ids)}")

    root = Path(__file__).resolve().parents[1]
    cache_dir = args.cache_dir or (root / "outputs/detection_time_cache")
    output_dir = args.output_dir or (
        root / "outputs/experiments"
        / f"{args.catalog.stem}_{n_train}train_{args.budget_hours:g}h"
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"Output directory already contains files: {output_dir}")

    start = perf_counter()
    times, cache_path, cache_hit = load_or_compute_times(
        candidates, args.catalog, root, cache_dir, args.workers, compute,
        recompute=args.recompute_detection_times, cache_only=args.cache_only,
    )

    records = [
        PlanetDetectionTime(entry.universe_id, entry.star_id, time_s)
        for entry, time_s in zip(candidates, times)
    ]

    print(
        f"Detection-time calculations: {perf_counter() - start:.2f} s",
        flush=True,
    )

    train_records = [record for record in records if record.universe_id < n_train]

    test_records = [record for record in records if record.universe_id >= n_train]

    strategies = {}
    for objective in ("E", "P"):
        start = perf_counter()
        result = greedy_optimize(
            records=train_records,
            star_ids=star_ids,
            universe_ids=train_ids,
            total_time_s=args.budget_hours * 3600.0,
            objective=objective,
        )
        strategies[objective] = result

        selected = {
            star_id for star_id, time_s in result.allocation_s.items() if time_s > 0
        }

        print(f"\n{objective}-OPTIMIZED", flush=True)
        print(f"Selected stars: {len(selected)}", flush=True)
        print(f"Used time: {result.used_time_s / 3600:.2f} h")
        print(f"E[N_det]: {result.mission_yield.expected_detections:.4f}")
        print(f"P(N_det >= 1): {result.mission_yield.probability_at_least_one:.4f}")
        print(
            f"Optimization time: {perf_counter() - start:.2f} s",
            flush=True,
        )

    e_result = strategies["E"]
    p_result = strategies["P"]

    e_stars = {s for s, t in e_result.allocation_s.items() if t > 0}
    p_stars = {s for s, t in p_result.allocation_s.items() if t > 0}

    print("\nTARGET-SELECTION COMPARISON")
    print(f"Both strategies: {len(e_stars & p_stars)} stars")
    print(f"Only E: {len(e_stars - p_stars)} stars")
    print(f"Only P: {len(p_stars - e_stars)} stars")

    print("\nCOMPLETE TIME-ALLOCATION COMPARISON")
    print(f"{'ID':>3} {'Star':<20} {'E [h]':>9} {'P [h]':>9} {'P-E [h]':>10}")

    all_selected = e_stars | p_stars

    # Sort by largest absolute difference in observing time.
    ordered_stars = sorted(
        all_selected,
        key=lambda s: abs(p_result.allocation_s[s] - e_result.allocation_s[s]),
        reverse=True,
    )

    total_increase_h = 0.0
    total_decrease_h = 0.0

    for star_id in ordered_stars:
        e_hours = e_result.allocation_s[star_id] / 3600.0
        p_hours = p_result.allocation_s[star_id] / 3600.0
        delta_h = p_hours - e_hours

        total_increase_h += max(delta_h, 0.0)
        total_decrease_h += max(-delta_h, 0.0)

        print(
            f"{star_id:3d} {star_names[star_id]:<20} "
            f"{e_hours:9.3f} {p_hours:9.3f} {delta_h:+10.3f}"
        )

    print("\nTIME REDISTRIBUTION")
    print(f"Total increases: {total_increase_h:.3f} h")
    print(f"Total decreases: {total_decrease_h:.3f} h")
    print(f"Net difference: {total_increase_h - total_decrease_h:+.3f} h")

    # Compare both strategies within the same Monte Carlo universes.
    assert e_result.mission_yield.universe_ids == p_result.mission_yield.universe_ids

    e_counts = np.asarray(e_result.mission_yield.detections_per_universe)
    p_counts = np.asarray(p_result.mission_yield.detections_per_universe)

    e_success = e_counts >= 1
    p_success = p_counts >= 1

    both = int(np.count_nonzero(e_success & p_success))
    only_e = int(np.count_nonzero(e_success & ~p_success))
    only_p = int(np.count_nonzero(~e_success & p_success))
    neither = int(np.count_nonzero(~e_success & ~p_success))

    n = len(e_counts)
    assert both + only_e + only_p + neither == n

    differences = p_success.astype(int) - e_success.astype(int)
    delta_p = float(np.mean(differences))

    # Paired bootstrap: resample universes, keeping E/P outcomes together.
    rng = np.random.default_rng(20261021)
    indices = rng.integers(0, n, size=(10000, n))
    bootstrap_deltas = differences[indices].mean(axis=1)
    ci_low, ci_high = np.quantile(bootstrap_deltas, [0.025, 0.975])

    # Exact McNemar test, using only discordant pairs.
    discordant = only_e + only_p
    mcnemar_p = binomtest(only_p, discordant, p=0.5).pvalue if discordant > 0 else 1.0

    print("\nPAIRED UNIVERSE OUTCOMES")
    print(f"Both succeed:    {both:3d}")
    print(f"Only E succeeds: {only_e:3d}")
    print(f"Only P succeeds: {only_p:3d}")
    print(f"Neither succeeds:{neither:3d}")
    print(f"Discordant pairs: {discordant}")
    print(f"Paired probability gain: {100 * delta_p:+.2f} pp")
    print(
        f"95% paired bootstrap interval: [{100 * ci_low:+.2f}, {100 * ci_high:+.2f}] pp"
    )
    print(f"Exact McNemar p-value (exploratory): {mcnemar_p:.4f}")

    print("\nYIELD COMPARISON")
    print(
        "P advantage: "
        f"{100 * (p_result.mission_yield.probability_at_least_one - e_result.mission_yield.probability_at_least_one):.2f}"
        " percentage points"
    )
    print(
        "E advantage: "
        f"{e_result.mission_yield.expected_detections - p_result.mission_yield.expected_detections:.4f}"
        " detections"
    )

    if len(test_ids) > 0:
        e_test = evaluate_fixed_allocation(
            test_records,
            e_result.allocation_s,
            test_ids,
        )
        p_test = evaluate_fixed_allocation(
            test_records,
            p_result.allocation_s,
            test_ids,
        )

        e_success = np.asarray(e_test.detections_per_universe) >= 1
        p_success = np.asarray(p_test.detections_per_universe) >= 1

        print("\nINDEPENDENT EVALUATION")
        print(f"Evaluation universes: {len(test_ids)}")
        print(f"E-optimized: E[N] = {e_test.expected_detections:.4f}")
        print(f"E-optimized: P(N>=1) = {e_test.probability_at_least_one:.4f}")
        print(f"P-optimized: E[N] = {p_test.expected_detections:.4f}")
        print(f"P-optimized: P(N>=1) = {p_test.probability_at_least_one:.4f}")

        print("\nEVALUATION PAIRED OUTCOMES")
        print(f"Both succeed:     {np.count_nonzero(e_success & p_success)}")
        print(f"Only E succeeds:  {np.count_nonzero(e_success & ~p_success)}")
        print(f"Only P succeeds:  {np.count_nonzero(~e_success & p_success)}")
        print(f"Neither succeeds: {np.count_nonzero(~e_success & ~p_success)}")

        delta_p_test = p_test.probability_at_least_one - e_test.probability_at_least_one
        print(f"Evaluation P advantage: {100 * delta_p_test:+.2f} pp")

    save_experiment(
        output_dir=output_dir,
        catalog=args.catalog,
        cache_path=cache_path,
        cache_hit=cache_hit,
        universe_count=args.universes,
        n_train=n_train,
        budget_hours=args.budget_hours,
        candidates_count=len(candidates),
        star_ids=star_ids,
        star_names=star_names,
        strategies=strategies,
        evaluations={"E": e_test, "P": p_test} if len(test_ids) else None,
    )


if __name__ == "__main__":
    main()
