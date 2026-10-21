"""Conservative habitable-zone boundaries from Kopparapu et al. (2014)."""

from math import isfinite


def conservative_hz_flux_bounds(temperature_k: float) -> tuple[float, float]:
    """Return (outer, inner) stellar flux limits in Earth-flux units.

    Uses the 1-Earth-mass maximum-greenhouse and runaway-greenhouse
    boundaries. Valid for main-sequence stars from 2600 to 7200 K.
    """
    if not isfinite(temperature_k) or not 2600 <= temperature_k <= 7200:
        raise ValueError("Stellar temperature must be 2600–7200 K")

    t = temperature_k - 5780.0

    outer = 0.356 + 6.171e-5 * t + 1.698e-9 * t**2 - 3.198e-12 * t**3 - 5.575e-16 * t**4
    inner = 1.107 + 1.332e-4 * t + 1.580e-8 * t**2 - 8.308e-12 * t**3 - 1.931e-15 * t**4

    return outer, inner


def in_conservative_hz(
    incident_flux_searth: float | None,
    stellar_temperature_k: float,
) -> bool:
    """Determine whether a planet lies inside the conservative HZ."""
    if (
        incident_flux_searth is None
        or not isfinite(incident_flux_searth)
        or incident_flux_searth <= 0
    ):
        raise ValueError("A positive, finite incident stellar flux is required")

    outer, inner = conservative_hz_flux_bounds(stellar_temperature_k)
    return outer <= incident_flux_searth <= inner
