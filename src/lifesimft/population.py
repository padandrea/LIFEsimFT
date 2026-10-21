"""Import P-Pop planet populations into LIFEsimFT."""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lifesimft.constants import (
    ARCSEC_TO_RAD,
    OBLIQUITY_RAD,
    PARSEC_M,
    R_EARTH_M,
    R_SUN_M,
    T_SUN_K,
)
from lifesimft.detectability import Planet, Star


@dataclass(frozen=True)
class PopulationEntry:
    universe_id: int
    star_id: int
    star_name: str
    star: Star
    planet: Planet
    incident_flux_searth: float | None = None


def load_ppop(path: str | Path) -> list[PopulationEntry]:
    """Load P-Pop's two-header-line text format."""

    entries = []

    with open(path, encoding="utf-8") as file:
        next(file)  # Original column names
        reader = csv.DictReader(file, delimiter="\t")

        for row in reader:
            distance_m = float(row["Ds"]) * PARSEC_M
            stellar_radius = float(row["Rs"]) * R_SUN_M
            stellar_temperature = float(row["Ts"])

            luminosity = (stellar_radius / R_SUN_M) ** 2 * (
                stellar_temperature / T_SUN_K
            ) ** 4

            # Equatorial coordinates -> ecliptic latitude
            ra = np.radians(float(row["RA"]))
            dec = np.radians(float(row["Dec"]))

            sin_beta = np.sin(dec) * np.cos(OBLIQUITY_RAD) - np.cos(dec) * np.sin(
                OBLIQUITY_RAD
            ) * np.sin(ra)
            beta = np.arcsin(np.clip(sin_beta, -1.0, 1.0))

            star = Star(
                temperature_k=stellar_temperature,
                radius_m=stellar_radius,
                distance_m=distance_m,
                luminosity_lsun=luminosity,
                ecliptic_latitude_rad=float(beta),
                zodi_level=float(row["z"]),
            )

            # Projected angular separation -> physical sky-plane offset
            theta_rad = float(row["AngSep"]) * ARCSEC_TO_RAD

            planet = Planet(
                temperature_k=float(row["Tp"]),
                radius_m=float(row["Rp"]) * R_EARTH_M,
                position_m=(theta_rad * distance_m, 0.0),
            )

            entries.append(
                PopulationEntry(
                    universe_id=int(row["Nuniverse"]),
                    star_id=int(row["Nstar"]),
                    star_name=row["name"],
                    star=star,
                    planet=planet,
                    incident_flux_searth=(float(row["Fp"]) if row.get("Fp") else None),
                )
            )

    return entries
