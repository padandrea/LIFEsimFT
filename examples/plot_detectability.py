"""Detection-time sensitivity for an Earth-like planet."""

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lifesimft import reference as ref
from lifesimft.detectability import Planet, Star, calculate_detection_time


def main():
    star = Star(
        temperature_k=ref.STAR_TEMPERATURE_K,
        radius_m=ref.STAR_RADIUS_M,
        distance_m=ref.DISTANCE_M,
        luminosity_lsun=ref.STAR_LUMINOSITY_LSUN,
        ecliptic_latitude_rad=ref.ECLIPTIC_LATITUDE_RAD,
        zodi_level=ref.ZODI_LEVEL,
    )

    planet = Planet(
        temperature_k=ref.PLANET_TEMPERATURE_K,
        radius_m=ref.PLANET_RADIUS_M,
        position_m=(ref.PLANET_SEPARATION_M, 0.0),
    )

    def detection_hours(s, p):
        return calculate_detection_time(s, p, snr_threshold=5.0) / 3600.0

    # 1. Distance sweep (physical separation fixed at 1 AU)
    distances = np.array([5, 7.5, 10, 12.5, 15, 20])
    t_distance = [
        detection_hours(
            replace(star, distance_m=star.distance_m * d / 10),
            planet,
        )
        for d in distances
    ]

    # 2. Planet radius sweep
    radii = np.array([0.5, 0.75, 1, 1.25, 1.5, 2])
    t_radius = [
        detection_hours(
            star,
            replace(planet, radius_m=planet.radius_m * r),
        )
        for r in radii
    ]

    # 3. Planet temperature sweep
    temperatures = np.array([180, 200, 220, 254, 280, 300])
    t_temperature = [
        detection_hours(
            star,
            replace(planet, temperature_k=float(temp)),
        )
        for temp in temperatures
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for ax, x, y, xlabel in zip(
        axes,
        [distances, radii, temperatures],
        [t_distance, t_radius, t_temperature],
        ["Distance (pc)", "Planet radius (Earth radii)", "Temperature (K)"],
    ):
        ax.plot(x, y, "o-")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Detection time (h)")
        ax.grid(True, alpha=0.3)

    fig.suptitle("LIFEsimFT — SNR = 5 detection threshold")
    fig.tight_layout()

    Path("outputs").mkdir(exist_ok=True)
    fig.savefig("outputs/detectability_sweeps.png", dpi=200)

    print(f"Reference detection time: {detection_hours(star, planet):.2f} h")
    plt.show()


if __name__ == "__main__":
    main()
