"""Benchmark parallel LIFEsimFT planet detection-time calculations.

Run from a terminal in the active lifesimft environment, not from a notebook:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python examples/benchmark_detection_workers.py

Uses the existing SAG13 100-universe dataset for a *performance benchmark only*.
This benchmark's radius/temperature filter does not define a LIFE habitable zone.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

from lifesimft.constants import R_EARTH_M
from lifesimft.detectability import calculate_detection_time
from lifesimft.population import PopulationEntry, load_ppop


def detection_time(entry: PopulationEntry) -> float:
    """Top-level picklable function for macOS spawn-based worker processes."""
    return calculate_detection_time(entry.star, entry.planet, snr_threshold=5.0)


def compute(
    entries: list[PopulationEntry], workers: int, chunksize: int
) -> list[float]:
    """Return detection times in input order."""
    if workers == 1:
        return [detection_time(entry) for entry in entries]
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("spawn"),
    ) as pool:
        return list(pool.map(detection_time, entries, chunksize=chunksize))


def matching(baseline: list[float], other: list[float]) -> bool:
    if len(baseline) != len(other):
        return False
    return all(
        (a == b)
        or (
            math.isfinite(a)
            and math.isfinite(b)
            and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-8)
        )
        for a, b in zip(baseline, other)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path.home() / "Developer/P-pop/YieldPopulation100.txt",
        help="P-Pop generated planet table",
    )
    parser.add_argument("--targets", type=int, default=10)
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 4, 8, 12, 18])
    parser.add_argument("--chunksize", type=int, default=8)
    parser.add_argument(
        "--max-planets",
        type=int,
        default=None,
        help="Optional first N eligible entries; use full population by default",
    )
    args = parser.parse_args()
    if args.targets <= 0 or args.chunksize <= 0:
        parser.error("--targets and --chunksize must be positive")
    if not args.workers or any(n <= 0 for n in args.workers):
        parser.error("Worker counts must be positive")
    if args.workers[0] != 1:
        parser.error("List workers beginning with 1 for a serial reference")

    start_load = perf_counter()
    population = load_ppop(args.catalog)
    entries = [
        item
        for item in population
        if item.star_id < args.targets
        and 0.5 <= item.planet.radius_m / R_EARTH_M <= 1.5
        and 200.0 <= item.planet.temperature_k <= 320.0
    ]
    if args.max_planets is not None:
        entries = entries[: args.max_planets]
    print(f"Catalog: {args.catalog}", flush=True)
    print(f"Eligible planets: {len(entries)}", flush=True)
    print(f"Catalog read/filter: {perf_counter() - start_load:.2f} s", flush=True)
    if not entries:
        parser.error("No candidates found; check catalog and target filter")

    baseline: list[float] | None = None
    baseline_time = 0.0
    print("\nWorkers    Seconds   Speedup  Results match", flush=True)
    for workers in args.workers:
        start = perf_counter()
        times = compute(entries, workers, args.chunksize)
        elapsed = perf_counter() - start
        if baseline is None:
            baseline = times
            baseline_time = elapsed
        same = matching(baseline, times)
        print(
            f"{workers:7d} {elapsed:10.2f} {baseline_time / elapsed:9.2f}x "
            f"{'yes' if same else 'NO'}",
            flush=True,
        )
        if not same:
            raise RuntimeError(f"Detection times changed with {workers} workers")

    print("\nAll worker counts agree numerically with the serial baseline.")
    print("Note: worker startup is included in each measured runtime.")


if __name__ == "__main__":
    main()
