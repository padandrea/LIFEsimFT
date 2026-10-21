"""Tests for Kopparapu conservative habitable-zone selection."""

import pytest

from lifesimft.habitable_zone import (
    conservative_hz_flux_bounds,
    in_conservative_hz,
)


def test_solar_hz_boundaries():
    outer, inner = conservative_hz_flux_bounds(5780.0)
    assert outer == pytest.approx(0.356)
    assert inner == pytest.approx(1.107)


def test_planets_inside_and_outside_hz():
    assert in_conservative_hz(1.0, 5780.0)
    assert in_conservative_hz(0.356, 5780.0)
    assert in_conservative_hz(1.107, 5780.0)
    assert not in_conservative_hz(0.2, 5780.0)
    assert not in_conservative_hz(1.5, 5780.0)


def test_stellar_temperature_changes_boundaries():
    cool = conservative_hz_flux_bounds(3900.0)
    solar = conservative_hz_flux_bounds(5780.0)
    hot = conservative_hz_flux_bounds(6300.0)

    assert cool[0] < solar[0] < hot[0]
    assert cool[1] < solar[1] < hot[1]


@pytest.mark.parametrize("temperature", [2500.0, 7300.0, float("nan")])
def test_invalid_stellar_temperature(temperature):
    with pytest.raises(ValueError):
        conservative_hz_flux_bounds(temperature)


@pytest.mark.parametrize("flux", [None, -1.0, float("nan")])
def test_invalid_planet_flux(flux):
    with pytest.raises(ValueError):
        in_conservative_hz(flux, 5780.0)
