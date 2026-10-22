"""Generate a reproducible P-Pop catalog using parallel independent batches.

The existing P-pop/P-pop.py provides the model configuration. This driver does
not modify it: separate processes run the same code with a distinct output
name, universe count, and NumPy random seed for each batch.

The merged output keeps P-Pop's original two header lines, and remaps universe
IDs to 0..N-1. Fixed batch boundaries/seeds mean worker count does not change
the simulated population.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import numpy as np

ORIGINAL_NAME = "Name = 'LTC4_BrysonLow_100'"
ORIGINAL_COUNT = "Nuniverses = 100  # int"
ENTRY_POINT = "SysGen = SystemGenerator.SystemGenerator("


def _replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"Expected one occurrence of {old!r} in P-pop.py")
    return source.replace(old, new, 1)


def _generate_batch(
    template: str,
    module_dir: Path,
    scratch: Path,
    index: int,
    count: int,
    seed: int,
) -> Path:
    name = f"batch_{index:04d}"
    batch_dir = scratch / name
    batch_dir.mkdir()

    code = _replace_once(template, ORIGINAL_NAME, f"Name = '{name}'")
    code = _replace_once(code, ORIGINAL_COUNT, f"Nuniverses = {count}  # int")
    code = _replace_once(
        code,
        ENTRY_POINT,
        f"import numpy as np\nnp.random.seed({seed})\n" + ENTRY_POINT,
    )
    # `python -c` has no script directory in sys.path; include P-Pop's modules.
    code = f"import sys\nsys.path.insert(0, {str(module_dir)!r})\n" + code

    env = os.environ.copy()
    env.update(
        {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=batch_dir,
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(
            f"P-Pop batch {index} failed (exit {result.returncode}).\n"
            f"stdout:\n{result.stdout[-2500:]}\n"
            f"stderr:\n{result.stderr[-4000:]}"
        )
    catalog = batch_dir / f"{name}.txt"
    if not catalog.is_file():
        raise RuntimeError(f"P-Pop batch {index} produced no catalog: {catalog}")
    return catalog


def _merge_catalogs(
    batch_files: list[Path],
    batch_sizes: list[int],
    output: Path,
) -> tuple[int, int]:
    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"Remove incomplete previous output: {partial}")

    headers = None
    planets = 0
    universe_start = 0
    try:
        with partial.open("x", encoding="utf-8") as merged:
            for catalog, n_universes in zip(batch_files, batch_sizes):
                with catalog.open(encoding="utf-8") as stream:
                    local_headers = (stream.readline(), stream.readline())
                    if not all(local_headers):
                        raise ValueError(f"Incomplete headers in {catalog}")
                    if headers is None:
                        headers = local_headers
                        merged.writelines(headers)
                    elif local_headers != headers:
                        raise ValueError(f"Header mismatch in {catalog}")
                    for line in stream:
                        if not line.strip():
                            continue
                        first, separator, remainder = line.partition("\t")
                        if not separator:
                            raise ValueError(f"Unexpected P-Pop row in {catalog}")
                        original_id = int(first)
                        if not 0 <= original_id < n_universes:
                            raise ValueError(
                                f"Universe ID {original_id} outside [0, {n_universes})"
                            )
                        merged.write(f"{universe_start + original_id}\t{remainder}")
                        planets += 1
                universe_start += n_universes
        partial.replace(output)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return universe_start, planets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universes", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument(
        "--ppop-root", type=Path, default=Path.home() / "Developer/P-pop"
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.universes < 1 or args.workers < 1 or args.batch_size < 1:
        parser.error("--universes, --workers, and --batch-size must be positive")

    root = args.ppop_root.expanduser().resolve()
    module_dir = root / "P-pop"
    template_path = module_dir / "P-pop.py"
    template = template_path.read_text(encoding="utf-8")
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else root / f"LTC4_BrysonLow_{args.universes}.txt"
    )
    if output.exists():
        parser.error(f"Refusing to overwrite existing catalog: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    sizes = [
        min(args.batch_size, args.universes - start)
        for start in range(0, args.universes, args.batch_size)
    ]
    seeds = [
        int(s.generate_state(1, dtype=np.uint32)[0])
        for s in np.random.SeedSequence(args.seed).spawn(len(sizes))
    ]
    print(
        f"Generating {args.universes} universes in {len(sizes)} batches, "
        f"up to {min(args.workers, len(sizes))} parallel workers",
        flush=True,
    )
    start_time = perf_counter()
    with tempfile.TemporaryDirectory(prefix="ppop_batches_") as temp:
        scratch = Path(temp)
        batch_files: list[Path | None] = [None] * len(sizes)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    _generate_batch, template, module_dir, scratch, i, n, seeds[i]
                ): i
                for i, n in enumerate(sizes)
            }
            for finished, future in enumerate(as_completed(futures), 1):
                index = futures[future]
                batch_files[index] = future.result()
                print(
                    f"Completed batch {finished}/{len(sizes)} "
                    f"({perf_counter() - start_time:.1f} s elapsed)",
                    flush=True,
                )
        assert all(path is not None for path in batch_files)
        n_merged, n_planets = _merge_catalogs(batch_files, sizes, output)

    assert n_merged == args.universes
    print(f"Saved: {output}")
    print(f"Universes: {n_merged}; planets: {n_planets}")
    print(f"Total runtime: {perf_counter() - start_time:.1f} s")


if __name__ == "__main__":
    main()
