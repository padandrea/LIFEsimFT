import numpy as np
import pytest

from lifesimft.statistics import planet_template, test_statistic as compute_t

SIGNAL = np.sin(np.linspace(0, 6 * np.pi, 600, endpoint=False))


def test_template_has_unit_rms():
    eta = planet_template(3.7 * SIGNAL)
    assert np.sqrt(np.mean(eta**2)) == pytest.approx(1.0)


def test_t_scales_with_square_root_of_time():
    t1 = compute_t(SIGNAL, 10.0, 10.0, total_time=100.0)
    t4 = compute_t(SIGNAL, 10.0, 10.0, total_time=400.0)
    assert t4 == pytest.approx(2 * t1)


def test_t_matches_poisson_monte_carlo():
    rng = np.random.default_rng(1)
    n_samples, total_time = 600, 6000.0
    dt = total_time / n_samples
    leakage = 5.0
    planet = 0.5 * SIGNAL
    rate_left = leakage + np.clip(planet, 0, None)
    rate_right = leakage + np.clip(-planet, 0, None)
    diff = rate_left - rate_right
    expected = compute_t(diff, rate_left.mean(), rate_right.mean(), total_time)
    counts = rng.poisson(rate_left * dt, (4000, n_samples)) - rng.poisson(
        rate_right * dt, (4000, n_samples)
    )
    filtered = counts @ planet_template(diff)
    assert filtered.mean() / filtered.std() == pytest.approx(expected, rel=0.05)