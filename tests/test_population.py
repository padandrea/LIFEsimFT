"""Tests for the P-Pop population importer."""

import pytest
from scipy.constants import au, parsec

from lifesimft.constants import R_EARTH_M, R_SUN_M
from lifesimft.population import load_ppop


@pytest.fixture
def sample_catalog(tmp_path):
    """Minimal P-Pop catalog with two Monte Carlo universes."""
    columns = [
        "Nuniverse",
        "Nstar",
        "name",
        "Ds",
        "Rs",
        "Ts",
        "z",
        "RA",
        "Dec",
        "Tp",
        "Rp",
        "AngSep",
    ]

    rows = [
        ["0", "11", "TestStar", "10", "1", "5778", "1", "0", "0", "254", "1", "0.1"],
        ["1", "11", "TestStar", "10", "1", "5778", "1", "0", "0", "200", "0.5", "0.2"],
    ]

    path = tmp_path / "population.txt"
    path.write_text(
        "Original P-Pop header\n"
        + "\t".join(columns)
        + "\n"
        + "\n".join("\t".join(row) for row in rows)
        + "\n"
    )
    return path


def test_population_entries(sample_catalog):
    """Both universes must be preserved."""
    population = load_ppop(sample_catalog)

    assert len(population) == 2
    assert {p.universe_id for p in population} == {0, 1}
    assert {p.star_id for p in population} == {11}
    assert population[0].star_name == "TestStar"


def test_unit_conversion(sample_catalog):
    """Astronomical units must be converted into SI."""
    entry = load_ppop(sample_catalog)[0]

    assert entry.star.distance_m == pytest.approx(10 * parsec)
    assert entry.star.radius_m == pytest.approx(R_SUN_M)
    assert entry.planet.radius_m == pytest.approx(R_EARTH_M)
    assert entry.star.luminosity_lsun == pytest.approx(1.0)


def test_projected_separation(sample_catalog):
    """0.1 arcsec at 10 pc corresponds to 1 AU."""
    entry = load_ppop(sample_catalog)[0]

    assert entry.planet.position_m[0] == pytest.approx(au, rel=1e-5)
    assert entry.planet.position_m[1] == 0.0


def test_ecliptic_latitude(sample_catalog):
    """RA = 0, Dec = 0 corresponds to ecliptic latitude 0."""
    entry = load_ppop(sample_catalog)[0]

    assert entry.star.ecliptic_latitude_rad == pytest.approx(0.0)
