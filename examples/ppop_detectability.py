"""Calculate detection times for a small P-Pop sample."""

from pathlib import Path
from time import perf_counter

from lifesimft.constants import R_EARTH_M
from lifesimft.detectability import calculate_detection_time
from lifesimft.population import load_ppop


def main():
    path = Path.home() / "Developer/P-pop/TestPlanetPopulation.txt"
    population = load_ppop(path)

    # Preliminary Earth-like candidates in universe 0.
    # This is NOT a formal habitable-zone selection.
    candidates = [
        entry
        for entry in population
        if entry.universe_id == 0
        and 0.5 <= entry.planet.radius_m / R_EARTH_M <= 1.5
        and 200 <= entry.planet.temperature_k <= 320
    ]

    # Select at most one planet per host star.
    selected = list({p.star_id: p for p in candidates}.values())[:5]

    print(f"Total planets: {len(population)}")
    print(f"Preliminary candidates in universe 0: {len(candidates)}")
    print()

    start = perf_counter()

    for entry in selected:
        t_det = calculate_detection_time(
            entry.star,
            entry.planet,
            snr_threshold=5.0,
        )

        print(
            f"{entry.star_name:15s} "
            f"T={entry.planet.temperature_k:6.1f} K  "
            f"R={entry.planet.radius_m / R_EARTH_M:.2f} R_earth  "
            f"t_det={t_det / 3600:.2f} h"
        )

    elapsed = perf_counter() - start

    print(f"\nComputed {len(selected)} planets in {elapsed:.2f} s")
    if selected:
        print(f"Average: {elapsed / len(selected):.3f} s/planet")


if __name__ == "__main__":
    main()
