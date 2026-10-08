import numpy as np
import pytest

from lifesimft import reference as ref
from lifesimft.geometry import baselines, rotate_positions, rotation_angles

ATOL_M = 1e-9


@pytest.fixture
def rotated_positions() -> np.ndarray:
    angles = rotation_angles(ref.N_ROTATIONS, ref.N_SAMPLES)
    return rotate_positions(ref.COLLECTOR_POSITIONS_M, angles)


def test_output_shapes(rotated_positions):
    assert rotated_positions.shape == (ref.N_SAMPLES, 4, 2)
    assert baselines(rotated_positions).shape == (ref.N_SAMPLES, 4, 4, 2)


def test_baselines_antisymmetric(rotated_positions):
    b = baselines(rotated_positions)
    np.testing.assert_allclose(b, -np.swapaxes(b, 1, 2), atol=ATOL_M)


def test_baseline_diagonal_is_zero(rotated_positions):
    b = baselines(rotated_positions)
    idx = np.arange(4)
    np.testing.assert_allclose(b[:, idx, idx], 0.0, atol=ATOL_M)


def test_baseline_lengths_constant_under_rotation(rotated_positions):
    lengths = np.linalg.norm(baselines(rotated_positions), axis=-1)
    np.testing.assert_allclose(lengths - lengths[0], 0.0, atol=ATOL_M)


def test_reference_baselines(rotated_positions):
    b = baselines(rotated_positions)[0]
    assert np.linalg.norm(b[0, 2]) == pytest.approx(14.5)
    assert np.linalg.norm(b[0, 1]) == pytest.approx(87.0)


def test_initial_positions_match_table_1(rotated_positions):
    np.testing.assert_allclose(
        rotated_positions[0], ref.COLLECTOR_POSITIONS_M, atol=ATOL_M
    )


def test_quarter_rotation_is_counter_clockwise():
    positions = ref.COLLECTOR_POSITIONS_M
    rotated = rotate_positions(positions, rotation_angles(1, 4))
    expected = np.column_stack([-positions[:, 1], positions[:, 0]])
    np.testing.assert_allclose(rotated[1], expected, atol=ATOL_M)


def test_angles_exclude_endpoint():
    angles = rotation_angles(ref.N_ROTATIONS, ref.N_SAMPLES)
    assert angles[0] == 0.0
    assert angles[-1] < 2 * np.pi * ref.N_ROTATIONS


def test_reference_exposure_time():
    assert ref.TOTAL_TIME_S / ref.N_SAMPLES == pytest.approx(594.58, abs=0.01)
