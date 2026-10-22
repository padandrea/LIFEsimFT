"""Unit tests for persistent detection-time caching and E/P summaries."""

import numpy as np
import pytest

from lifesimft.experiment_io import load_or_compute_times, paired_statistics


class Candidate:
    def __init__(self, universe_id, star_id):
        self.universe_id = universe_id
        self.star_id = star_id

    def __repr__(self):
        return f"Candidate({self.universe_id}, {self.star_id})"


def test_cache_creation_reuse_and_population_invalidation(tmp_path):
    root = tmp_path / "repo"
    (root / "src/lifesimft").mkdir(parents=True)
    (root / "examples").mkdir()
    (root / "src/lifesimft/physics.py").write_text("version = 1\n")
    (root / "examples/benchmark_detection_workers.py").write_text("# compute\n")
    catalog = tmp_path / "catalog.txt"
    catalog.write_text("version1\n")
    entries = [Candidate(0, 1), Candidate(1, 1)]
    calls = []

    def compute(items, workers, chunksize):
        calls.append((len(items), workers, chunksize))
        return [3600.0, float("inf")]

    kwargs = dict(candidates=entries, catalog=catalog, root=root,
                  cache_dir=tmp_path / "cache", workers=2, compute=compute)
    times, cache_path, reused = load_or_compute_times(**kwargs)
    assert np.array_equal(times, [3600.0, float("inf")])
    assert cache_path.exists()
    assert reused is False
    assert len(calls) == 1

    again, same_path, reused = load_or_compute_times(**kwargs, cache_only=True)
    assert same_path == cache_path
    assert np.array_equal(again, times)
    assert reused is True
    assert len(calls) == 1

    catalog.write_text("version2\n")
    with pytest.raises(FileNotFoundError):
        load_or_compute_times(**kwargs, cache_only=True)
    new_times, new_path, reused = load_or_compute_times(**kwargs)
    assert new_path != cache_path
    assert reused is False
    assert len(calls) == 2
    assert np.array_equal(new_times, times)


def test_corrupt_cache_fails_closed(tmp_path):
    root = tmp_path / "repo"
    (root / "src/lifesimft").mkdir(parents=True)
    (root / "examples").mkdir()
    (root / "src/lifesimft/physics.py").write_text("x\n")
    (root / "examples/benchmark_detection_workers.py").write_text("x\n")
    catalog = tmp_path / "catalog.txt"
    catalog.write_text("x\n")
    kwargs = dict(candidates=[Candidate(0, 0)], catalog=catalog, root=root,
                  cache_dir=tmp_path / "cache", workers=1,
                  compute=lambda entries, workers, chunksize: [3.0])
    _, cache, _ = load_or_compute_times(**kwargs)
    cache.write_bytes(b"not an npz")
    with pytest.raises(RuntimeError, match="Invalid cache"):
        load_or_compute_times(**kwargs)


def test_paired_statistics():
    result = paired_statistics([1, 1, 0, 0], [1, 0, 1, 0], seed=42)
    assert result["both"] == 1
    assert result["only_e"] == 1
    assert result["only_p"] == 1
    assert result["neither"] == 1
    assert result["p_minus_e"] == pytest.approx(0.0)
    assert result["mcnemar_exact_p"] == pytest.approx(1.0)
    assert len(result["bootstrap_95_interval"]) == 2
