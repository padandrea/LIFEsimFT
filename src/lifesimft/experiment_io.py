"""Persist HZ survey simulations and reuse validated detection-time calculations."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_fingerprint(catalog: Path, candidates: list, root: Path) -> str:
    """Hash population, filtered planets, and detector/physics implementation."""
    digest = hashlib.sha256(b"LIFEsimFT-detection-cache-v1\n")
    digest.update(sha256_file(catalog).encode())
    # Only detection/selection code affects the threshold cache. Changes to
    # result serialization and greedy scheduling must not invalidate expensive
    # planet detection-time calculations.
    excluded = {"experiment_io.py", "mission_yield.py", "optimization.py"}
    source_files = sorted(
        path for path in (root / "src" / "lifesimft").rglob("*.py")
        if path.name not in excluded
    )
    source_files.append(root / "examples" / "benchmark_detection_workers.py")
    for path in source_files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(sha256_file(path).encode())
    for entry in candidates:
        digest.update(repr(entry).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_or_compute_times(
    candidates: list,
    catalog: Path,
    root: Path,
    cache_dir: Path,
    workers: int,
    compute,
    *,
    recompute: bool = False,
    cache_only: bool = False,
):
    """Load a matching cache or calculate and atomically save exact threshold times."""
    if recompute and cache_only:
        raise ValueError("Cannot combine recompute and cache-only")
    key = cache_fingerprint(catalog, candidates, root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"hz_detection_times_{key}.npz"
    universe_ids = np.asarray([item.universe_id for item in candidates], dtype=np.int64)
    star_ids = np.asarray([item.star_id for item in candidates], dtype=np.int64)

    if path.exists() and not recompute:
        try:
            with np.load(path, allow_pickle=False) as saved:
                if saved["fingerprint"].item() != key:
                    raise ValueError("Cache fingerprint mismatch")
                if not np.array_equal(saved["universe_id"], universe_ids):
                    raise ValueError("Cached universe IDs mismatch")
                if not np.array_equal(saved["star_id"], star_ids):
                    raise ValueError("Cached star IDs mismatch")
                times = np.asarray(saved["detection_time_s"], dtype=np.float64)
            if times.shape != (len(candidates),) or np.isnan(times).any() or (times < 0).any():
                raise ValueError("Invalid cached detection times")
        except (ValueError, KeyError, OSError) as exc:
            raise RuntimeError(f"Invalid cache {path}; use --recompute-detection-times") from exc
        print(f"Loaded {len(times)} detection times from {path}", flush=True)
        return times, path, True

    if cache_only:
        raise FileNotFoundError(f"No matching detection-time cache: {path}")
    times = np.asarray(compute(candidates, workers, chunksize=8), dtype=np.float64)
    if times.shape != (len(candidates),) or np.isnan(times).any() or (times < 0).any():
        raise ValueError("Computed detection times are invalid")
    temp_path = path.with_name(path.stem + ".tmp.npz")
    try:
        np.savez_compressed(
            temp_path,
            fingerprint=np.array(key),
            universe_id=universe_ids,
            star_id=star_ids,
            detection_time_s=times,
        )
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)
    print(f"Saved detection-time cache: {path}", flush=True)
    return times, path, False


def paired_statistics(e_counts, p_counts, *, seed: int) -> dict:
    """Paired bootstrap and exact McNemar test for fixed strategies."""
    e = np.asarray(e_counts) >= 1
    p = np.asarray(p_counts) >= 1
    if e.ndim != 1 or e.shape != p.shape or e.size == 0:
        raise ValueError("Expected aligned, non-empty per-universe counts")
    both = int(np.count_nonzero(e & p))
    only_e = int(np.count_nonzero(e & ~p))
    only_p = int(np.count_nonzero(~e & p))
    neither = int(np.count_nonzero(~e & ~p))
    diff = p.astype(np.int8) - e.astype(np.int8)
    rng = np.random.default_rng(seed)
    boot = np.empty(10000)
    for start in range(0, len(boot), 200):
        end = min(start + 200, len(boot))
        indices = rng.integers(0, diff.size, size=(end - start, diff.size))
        boot[start:end] = diff[indices].mean(axis=1)
    ci = np.quantile(boot, (0.025, 0.975))
    discordant = only_e + only_p
    return {
        "both": both,
        "only_e": only_e,
        "only_p": only_p,
        "neither": neither,
        "discordant": discordant,
        "p_minus_e": float(diff.mean()),
        "bootstrap_95_interval": [float(ci[0]), float(ci[1])],
        "mcnemar_exact_p": float(binomtest(only_p, discordant).pvalue)
        if discordant else 1.0,
        "bootstrap_resamples": 10000,
        "bootstrap_seed": seed,
    }


def _json_numpy_scalar(value):
    """Convert NumPy scalar values without silently coercing arrays."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def save_experiment(
    *,
    output_dir: Path,
    catalog: Path,
    cache_path: Path,
    cache_hit: bool,
    universe_count: int,
    n_train: int,
    budget_hours: float,
    candidates_count: int,
    star_ids,
    star_names,
    strategies,
    evaluations=None,
) -> None:
    """Save full-precision allocations, individual yields, and aggregate metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    train = {key: result.mission_yield for key, result in strategies.items()}
    splits = [("training", train)]
    if evaluations is not None:
        splits.append(("evaluation", evaluations))

    with (output_dir / "allocations.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["star_id", "star_name", "E_time_s", "P_time_s", "E_time_h", "P_time_h"])
        for star in sorted(star_ids):
            e = strategies["E"].allocation_s[star]
            p = strategies["P"].allocation_s[star]
            writer.writerow([star, star_names[star], repr(e), repr(p), repr(e / 3600), repr(p / 3600)])

    paired = {}
    with (output_dir / "per_universe.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "universe_id", "E_N_det", "P_N_det", "E_success", "P_success"])
        for split, results in splits:
            e = results["E"]
            p = results["P"]
            if e.universe_ids != p.universe_ids:
                raise ValueError("Mismatching E/P universe IDs")
            for uid, e_n, p_n in zip(
                e.universe_ids, e.detections_per_universe, p.detections_per_universe
            ):
                writer.writerow([split, uid, e_n, p_n, int(e_n >= 1), int(p_n >= 1)])
            paired[split] = paired_statistics(
                e.detections_per_universe,
                p.detections_per_universe,
                seed=20261021 if split == "training" else 20261022,
            )

    def statistics(yield_result):
        return {
            "expected_detections": yield_result.expected_detections,
            "probability_at_least_one": yield_result.probability_at_least_one,
        }

    summary = {
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "catalog": str(catalog.resolve()),
        "catalog_sha256": sha256_file(catalog),
        "cache_file": str(cache_path.resolve()),
        "cache_reused": cache_hit,
        "universe_count": universe_count,
        "training_count": n_train,
        "evaluation_count": universe_count - n_train,
        "eligible_planet_count": candidates_count,
        "budget_hours": budget_hours,
        "optimizer": "incremental greedy (not guaranteed globally optimal)",
        "training": {
            key: {
                **statistics(result.mission_yield),
                "used_time_hours": result.used_time_s / 3600,
                "selected_stars": sum(t > 0 for t in result.allocation_s.values()),
            }
            for key, result in strategies.items()
        },
        "evaluation": (
            {key: statistics(result) for key, result in evaluations.items()}
            if evaluations is not None else None
        ),
        "paired_statistics": paired,
    }
    temp = output_dir / "summary.json.tmp"
    try:
        temp.write_text(json.dumps(summary, indent=2, allow_nan=False, default=_json_numpy_scalar) + "\n", encoding="utf-8")
        temp.replace(output_dir / "summary.json")
    finally:
        temp.unlink(missing_ok=True)

    if "evaluation" in paired:
        result = paired["evaluation"]
        low, high = result["bootstrap_95_interval"]
        print(
            f"Holdout paired P-E 95% bootstrap interval: [{100 * low:+.2f}, {100 * high:+.2f}] pp"
        )
        print(f"Holdout McNemar exact p-value: {result['mcnemar_exact_p']:.4f}")
    print(f"Saved results: {output_dir.resolve()}", flush=True)
    print("Files: summary.json, allocations.csv, per_universe.csv", flush=True)
