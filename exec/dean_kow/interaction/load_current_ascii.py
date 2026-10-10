#!/usr/bin/env python3
"""Load AMReX current ASCII diagnostics into NumPy arrays.

Expected row formats:
  cell lev i j k phi_old phi_new
  face lev dir i j k integrated_current

By default, current arrays use the face coordinates that are actually printed
in the ASCII files. For periodic output produced with AMReX OwnerMask, this can
be Nx by Ny even though full x- and y-face grids would be Nx+1 by Ny and
Nx by Ny+1, respectively.
"""

from __future__ import annotations

import argparse
import glob
import math
import re
from pathlib import Path

import numpy as np


ParsedRow = tuple[str, int, str | None, int, int, int, float, float | None]


def _step_from_name(path: str | Path) -> int:
    """Use the last integer in the filename as the timestep sort key."""
    matches = re.findall(r"\d+", Path(path).name)
    return int(matches[-1]) if matches else -1


def _expand_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        paths.extend(Path(m) for m in matches)
    return sorted(set(paths), key=lambda p: (_step_from_name(p), str(p)))


def _parse_row(path: Path, line_number: int, line: str) -> ParsedRow | None:
    """Parse one data row, ignoring blank, comment, and unrelated rows."""
    fields = line.split()
    if not fields or fields[0].startswith("#"):
        return None

    kind = fields[0]
    if kind not in {"cell", "face"}:
        return None
    if len(fields) != 7:
        raise ValueError(
            f"{path}:{line_number}: expected 7 fields for a {kind} row, "
            f"got {len(fields)}"
        )

    try:
        level = int(fields[1])
        if kind == "cell":
            direction = None
            i, j, k = map(int, fields[2:5])
            first = float(fields[5])
            second = float(fields[6])
        else:
            direction = fields[2]
            i, j, k = map(int, fields[3:6])
            first = float(fields[6])
            second = None
    except ValueError as exc:
        raise ValueError(f"{path}:{line_number}: invalid {kind} row: {exc}") from exc

    return kind, level, direction, i, j, k, first, second


def _levels_in_file(path: Path) -> set[int]:
    levels: set[int] = set()
    with path.open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            row = _parse_row(path, line_number, line)
            if row is not None and row[0] == "cell":
                levels.add(row[1])
    return levels


def _total_particle_count_in_file(path: Path) -> float | None:
    prefix = "# total_particle_count:"
    value: float | None = None
    with path.open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped.startswith(prefix):
                continue
            raw_value = stripped[len(prefix) :].strip()
            try:
                parsed = float(raw_value)
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid total_particle_count "
                    f"{raw_value!r}"
                ) from exc
            if not math.isfinite(parsed) or parsed <= 0:
                raise ValueError(
                    f"{path}:{line_number}: total_particle_count must be "
                    f"finite and positive, got {parsed}"
                )
            if value is not None and parsed != value:
                raise ValueError(
                    f"{path}:{line_number}: conflicting total_particle_count "
                    f"values {value} and {parsed}"
                )
            value = parsed
    return value


def _resolve_total_particle_count(
    paths: list[Path], override: float | None
) -> float:
    if override is not None and (not math.isfinite(override) or override <= 0):
        raise ValueError("total_particle_count must be finite and positive")

    values = {
        value
        for path in paths
        if (value := _total_particle_count_in_file(path)) is not None
    }
    if len(values) > 1:
        raise ValueError(
            "input files contain inconsistent total_particle_count values: "
            f"{sorted(values)}"
        )

    recorded = next(iter(values), None)
    if override is not None and recorded is not None and override != recorded:
        raise ValueError(
            "total_particle_count override does not match the value recorded "
            f"in the input files ({override} != {recorded})"
        )
    if override is not None:
        return float(override)
    if recorded is not None:
        return recorded
    return math.nan


def _scan_extents(
    paths: list[Path], level: int
) -> tuple[int, int, int, int, int, list[int], dict[str, dict[str, object]]]:
    imin = jmin = None
    imax = jmax = None
    cell_counts: list[int] = []
    face_stats: dict[str, dict[str, object]] = {}
    k_values: set[int] = set()

    for path_index, path in enumerate(paths):
        cell_count = 0
        face_counts: dict[str, int] = {}
        seen_cells: set[tuple[int, int, int]] = set()
        seen_faces: set[tuple[str, int, int, int]] = set()

        with path.open(encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, start=1):
                row = _parse_row(path, line_number, line)
                if row is None:
                    continue
                kind, row_level, direction, i, j, k, _, _ = row
                if row_level != level:
                    continue

                if kind == "cell":
                    coordinate = (i, j, k)
                    if coordinate in seen_cells:
                        raise ValueError(
                            f"{path}:{line_number}: duplicate cell coordinate "
                            f"at level {level}: {coordinate}"
                        )
                    seen_cells.add(coordinate)
                    k_values.add(k)
                    imin = i if imin is None else min(imin, i)
                    imax = i if imax is None else max(imax, i)
                    jmin = j if jmin is None else min(jmin, j)
                    jmax = j if jmax is None else max(jmax, j)
                    cell_count += 1

                elif kind == "face":
                    assert direction is not None
                    coordinate = (direction, i, j, k)
                    if coordinate in seen_faces:
                        raise ValueError(
                            f"{path}:{line_number}: duplicate face coordinate "
                            f"at level {level}: {coordinate}"
                        )
                    seen_faces.add(coordinate)
                    k_values.add(k)
                    if direction not in face_stats:
                        face_stats[direction] = {
                            "imin": i,
                            "imax": i,
                            "jmin": j,
                            "jmax": j,
                            "counts": [0] * path_index,
                        }
                    stats = face_stats[direction]
                    stats["imin"] = min(int(stats["imin"]), i)
                    stats["imax"] = max(int(stats["imax"]), i)
                    stats["jmin"] = min(int(stats["jmin"]), j)
                    stats["jmax"] = max(int(stats["jmax"]), j)
                    face_counts[direction] = face_counts.get(direction, 0) + 1

        cell_counts.append(cell_count)
        for direction, stats in face_stats.items():
            stats["counts"].append(face_counts.get(direction, 0))

    if imin is None or jmin is None or imax is None or jmax is None:
        raise ValueError(f"no cell rows found for level {level}")
    if len(k_values) != 1:
        raise ValueError(
            f"level {level} contains multiple k planes {sorted(k_values)}; "
            "this loader only supports 2D data"
        )

    return imin, imax, jmin, jmax, next(iter(k_values)), cell_counts, face_stats


def _full_face_shape(direction: str, nx: int, ny: int) -> tuple[int, int]:
    if direction == "x":
        return nx + 1, ny
    if direction == "y":
        return nx, ny + 1
    raise ValueError(
        f"cannot create a padded 2D face grid for direction {direction!r}; "
        "expected 'x' or 'y'"
    )


def _infer_face_layout(
    direction: str,
    nx: int,
    ny: int,
    face_shape: tuple[int, int],
    counts: list[int],
) -> str:
    ni, nj = face_shape
    expected_rect_count = ni * nj
    count_set = set(counts)

    if len(count_set) > 1:
        return "inconsistent-counts"
    if count_set and next(iter(count_set)) != expected_rect_count:
        return "sparse-or-masked"

    if ni == nx and nj == ny:
        return "periodic-compact"
    if direction == "x" and ni == nx + 1 and nj == ny:
        return "x-staggered-full"
    if direction == "y" and ni == nx and nj == ny + 1:
        return "y-staggered-full"
    if ni == nx + 1 and nj == ny + 1:
        return "nodal-square-full"
    return "unknown"


def load_current_ascii(
    patterns: str | list[str],
    *,
    level: int | None = None,
    current_grid: str = "printed",
    total_particle_count: float | None = None,
) -> dict[str, np.ndarray]:
    """Load current ASCII files.

    Parameters
    ----------
    patterns:
        One glob or a list of globs/files, e.g. ``"plt_pp_*_current_ascii.dat"``.
    level:
        AMR level to load. If omitted, the highest level in the first file is used.
    current_grid:
        ``"printed"`` stores currents using exactly the face coordinates printed
        in the ASCII files. ``"padded"`` stores x currents in ``(Nx+1, Ny)`` and
        y currents in ``(Nx, Ny+1)``. A missing periodic endpoint is copied from
        its equivalent endpoint. Other absent entries remain NaN.
    total_particle_count:
        Number used to normalize phi. New ASCII files record it in their header.
        Supply this argument when loading legacy files without that header.
    """
    if isinstance(patterns, str):
        patterns = [patterns]

    paths = _expand_paths(patterns)
    if not paths:
        raise FileNotFoundError(f"no files matched: {patterns}")

    if level is None:
        levels = _levels_in_file(paths[0])
        if not levels:
            raise ValueError(f"no cell rows found in {paths[0]}")
        level = max(levels)

    if current_grid not in {"printed", "padded"}:
        raise ValueError("current_grid must be 'printed' or 'padded'")
    total_particle_count = _resolve_total_particle_count(
        paths, total_particle_count
    )

    imin, imax, jmin, jmax, k_index, cell_counts, face_stats = _scan_extents(
        paths, level
    )
    nx = imax - imin + 1
    ny = jmax - jmin + 1
    nt = len(paths)
    if not math.isfinite(total_particle_count):
        raise ValueError(
            "total_particle_count is required to convert phi to particles per "
            "cell; use --total-particle-count for legacy ASCII files"
        )
    average_particle_count_per_cell = total_particle_count / (nx * ny)

    density_old = np.full((nt, nx, ny), np.nan, dtype=float)
    density_new = np.full((nt, nx, ny), np.nan, dtype=float)
    currents: dict[str, np.ndarray] = {}
    current_origins: dict[str, tuple[int, int]] = {}
    current_layouts: dict[str, str] = {}
    current_printed_shapes: dict[str, tuple[int, int]] = {}

    for direction, stats in sorted(face_stats.items()):
        face_imin = int(stats["imin"])
        face_imax = int(stats["imax"])
        face_jmin = int(stats["jmin"])
        face_jmax = int(stats["jmax"])
        printed_shape = (face_imax - face_imin + 1, face_jmax - face_jmin + 1)
        counts = list(stats["counts"])

        current_printed_shapes[direction] = printed_shape
        current_layouts[direction] = _infer_face_layout(
            direction, nx, ny, printed_shape, counts
        )

        if current_grid == "printed":
            current_shape = printed_shape
            current_origin = (face_imin, face_jmin)
        else:
            current_shape = _full_face_shape(direction, nx, ny)
            current_origin = (imin, jmin)

        currents[direction] = np.full((nt, *current_shape), np.nan, dtype=float)
        current_origins[direction] = current_origin

    steps = np.array([_step_from_name(path) for path in paths], dtype=int)

    for t, path in enumerate(paths):
        with path.open(encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, start=1):
                row = _parse_row(path, line_number, line)
                if row is None:
                    continue
                kind, row_level, direction, raw_i, raw_j, _, first, second = row
                if row_level != level:
                    continue

                if kind == "cell":
                    i = raw_i - imin
                    j = raw_j - jmin
                    density_old[t, i, j] = first
                    assert second is not None
                    density_new[t, i, j] = second

                elif kind == "face":
                    assert direction is not None
                    if direction not in currents:
                        continue
                    origin_i, origin_j = current_origins[direction]
                    i = raw_i - origin_i
                    j = raw_j - origin_j
                    _, ni, nj = currents[direction].shape
                    if not (0 <= i < ni and 0 <= j < nj):
                        raise ValueError(
                            f"{path}:{line_number}: face coordinate "
                            f"({raw_i}, {raw_j}) is outside the {direction!r} "
                            f"current grid with origin {current_origins[direction]} "
                            f"and shape {(ni, nj)}"
                        )
                    currents[direction][t, i, j] = first

    if current_grid == "padded":
        for direction, arr in currents.items():
            if current_layouts[direction] != "periodic-compact":
                continue
            if current_origins[direction] != (imin, jmin):
                raise ValueError(
                    f"cannot reconstruct periodic {direction!r} endpoints: "
                    "face and cell origins differ"
                )
            if direction == "x":
                arr[:, nx, :] = arr[:, 0, :]
            elif direction == "y":
                arr[:, :, ny] = arr[:, :, 0]

    density_old *= average_particle_count_per_cell
    density_new *= average_particle_count_per_cell
    if nt > 1:
        if not np.all(np.diff(steps) == 1):
            raise ValueError(
                "density time series requires consecutive input timesteps, "
                f"got {steps.tolist()}"
            )
        if not np.allclose(
            density_old[1:],
            density_new[:-1],
            rtol=1.0e-12,
            atol=1.0e-12,
            equal_nan=True,
        ):
            raise ValueError(
                "density_old in each file must match density_new in the "
                "preceding file"
            )
    density = np.concatenate((density_old[:1], density_new), axis=0)
    density_steps = np.concatenate(([steps[0] - 1], steps))
    out: dict[str, np.ndarray] = {
        "steps": steps,
        "density_steps": density_steps,
        "density": density,
        "level": np.array(level, dtype=int),
        "k_index": np.array(k_index, dtype=int),
        "origin": np.array([imin, jmin], dtype=int),
        "cell_shape": np.array([nx, ny], dtype=int),
        "cell_count_per_step": np.array(cell_counts, dtype=int),
        "current_grid": np.array(current_grid),
        "total_particle_count": np.array(total_particle_count, dtype=float),
        "average_particle_count_per_cell": np.array(
            average_particle_count_per_cell, dtype=float
        ),
    }
    for direction, arr in currents.items():
        out[f"current_{direction}"] = arr
        out[f"current_{direction}_origin"] = np.array(
            current_origins[direction], dtype=int
        )
        out[f"current_{direction}_printed_shape"] = np.array(
            current_printed_shapes[direction], dtype=int
        )
        full_shape = (
            _full_face_shape(direction, nx, ny)
            if direction in {"x", "y"}
            else current_printed_shapes[direction]
        )
        out[f"current_{direction}_full_periodic_shape"] = np.array(
            full_shape, dtype=int
        )
        out[f"current_{direction}_layout"] = np.array(current_layouts[direction])
        out[f"current_{direction}_faces_per_step"] = np.array(
            face_stats[direction]["counts"], dtype=int
        )
    return out


def _canonical_periodic_current(
    arrays: dict[str, np.ndarray],
    direction: str,
    nx: int,
    ny: int,
) -> np.ndarray:
    key = f"current_{direction}"
    if key not in arrays:
        raise KeyError(f"missing current array {key!r}")

    current = np.asarray(arrays[key], dtype=float)
    if current.ndim != 3:
        raise ValueError(f"{key} must have shape (time, i, j)")

    cell_origin = np.asarray(arrays.get("origin", [0, 0]), dtype=int)
    face_origin = np.asarray(arrays.get(f"{key}_origin", cell_origin), dtype=int)
    if cell_origin.shape != (2,) or face_origin.shape != (2,):
        raise ValueError("cell and current origins must each contain two indices")

    if direction == "x":
        if current.shape[1] not in {nx, nx + 1} or current.shape[2] != ny:
            raise ValueError(
                f"current_x must have spatial shape ({nx}, {ny}) or "
                f"({nx + 1}, {ny}), got {current.shape[1:]}"
            )
        if current.shape[1] == nx + 1:
            if not np.allclose(
                current[:, 0, :], current[:, nx, :], equal_nan=True
            ):
                raise ValueError(
                    "current_x periodic endpoint faces do not have equal values"
                )
            current = current[:, :nx, :]
    elif direction == "y":
        if current.shape[1] != nx or current.shape[2] not in {ny, ny + 1}:
            raise ValueError(
                f"current_y must have spatial shape ({nx}, {ny}) or "
                f"({nx}, {ny + 1}), got {current.shape[1:]}"
            )
        if current.shape[2] == ny + 1:
            if not np.allclose(
                current[:, :, 0], current[:, :, ny], equal_nan=True
            ):
                raise ValueError(
                    "current_y periodic endpoint faces do not have equal values"
                )
            current = current[:, :, :ny]
    else:
        raise ValueError("direction must be 'x' or 'y'")

    i_indices = (cell_origin[0] + np.arange(nx) - face_origin[0]) % nx
    j_indices = (cell_origin[1] + np.arange(ny) - face_origin[1]) % ny
    return current[:, i_indices, :][:, :, j_indices]


def check_periodic_evolution(
    arrays: dict[str, np.ndarray],
    *,
    atol: float = 1.0e-8,
    rtol: float = 1.0e-10,
) -> dict[str, np.ndarray]:
    """Check particle conservation across every consecutive loaded step.

    The current at index t advances density[t] to density[t+1].
    Positive x and y currents point right and up, respectively. Residuals are
    reported in particles, not normalized density. This check assumes a full,
    uniform, two-dimensional periodic grid.
    """
    steps = np.asarray(arrays["steps"], dtype=int)
    density = np.asarray(arrays["density"], dtype=float)
    density_steps = np.asarray(arrays["density_steps"], dtype=int)
    if density.ndim != 3:
        raise ValueError("density must have shape (time, i, j)")
    nstates, nx, ny = density.shape
    ntransitions = nstates - 1
    if ntransitions < 1:
        raise ValueError("at least two density states are required")
    if steps.shape != (ntransitions,):
        raise ValueError("steps must contain one value per current transition")
    if density_steps.shape != (nstates,):
        raise ValueError("density_steps must contain one value per density state")
    if not np.all(np.isfinite(density)):
        raise ValueError(
            "density contains missing or non-finite values; "
            "a full periodic grid is required"
        )

    current_x = _canonical_periodic_current(arrays, "x", nx, ny)
    current_y = _canonical_periodic_current(arrays, "y", nx, ny)
    if (
        current_x.shape[0] != ntransitions
        or current_y.shape[0] != ntransitions
    ):
        raise ValueError("currents must contain one array per density transition")
    if not np.all(np.isfinite(current_x)) or not np.all(np.isfinite(current_y)):
        raise ValueError(
            "currents contain missing or non-finite values; "
            "a full periodic grid is required"
        )

    before = density[:-1]
    after = density[1:]
    before_sums = before.sum(axis=(1, 2))
    after_sums = after.sum(axis=(1, 2))
    observed_delta_particles = after - before

    transition_x = current_x
    transition_y = current_y
    expected_delta_particles = (
        transition_x
        - np.roll(transition_x, -1, axis=1)
        + transition_y
        - np.roll(transition_y, -1, axis=2)
    )
    continuity_residual = observed_delta_particles - expected_delta_particles

    continuity_ok = np.all(
        np.isclose(
            observed_delta_particles,
            expected_delta_particles,
            atol=atol,
            rtol=rtol,
        ),
        axis=(1, 2),
    )
    density_sum_ok = np.isclose(
        before_sums, after_sums, atol=atol, rtol=rtol
    )
    consecutive_ok = np.diff(density_steps) == 1

    return {
        "from_steps": density_steps[:-1],
        "to_steps": density_steps[1:],
        "passed": continuity_ok & density_sum_ok & consecutive_ok,
        "continuity_residual_particles": continuity_residual,
        "max_abs_continuity_residual": np.max(
            np.abs(continuity_residual), axis=(1, 2)
        ),
        "density_sum_change": after_sums - before_sums,
        "steps_are_consecutive": consecutive_ok,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load *_current_ascii.dat files into NumPy arrays."
    )
    parser.add_argument("files", nargs="+", help="input files or glob patterns")
    parser.add_argument("--level", type=int, default=None, help="AMR level to load")
    parser.add_argument(
        "--current-grid",
        choices=("printed", "padded"),
        default="printed",
        help=(
            "store currents on the printed face grid, or pad each direction to "
            "its full face-centered shape"
        ),
    )
    parser.add_argument(
        "--total-particle-count",
        type=float,
        help=(
            "phi normalization count; required for meaningful metadata when "
            "loading legacy files without a total_particle_count header"
        ),
    )
    parser.add_argument("--out", help="optional .npz output path")
    args = parser.parse_args()

    arrays = load_current_ascii(
        args.files,
        level=args.level,
        current_grid=args.current_grid,
        total_particle_count=args.total_particle_count,
    )

    print(f"steps: {arrays['steps'].shape}")
    print(f"density_steps: {arrays['density_steps'].shape}")
    print(f"density: {arrays['density'].shape}")
    print(f"k_index: {arrays['k_index']}")
    print(f"total_particle_count: {arrays['total_particle_count']}")
    print(
        "average_particle_count_per_cell: "
        f"{arrays['average_particle_count_per_cell']}"
    )
    print(f"cell_shape: {arrays['cell_shape']}")
    print(f"cell_count_per_step: {arrays['cell_count_per_step']}")

    current_keys = sorted(
        key
        for key, value in arrays.items()
        if key.startswith("current_") and getattr(value, "ndim", 0) == 3
    )
    for key in current_keys:
        print(f"{key}: {arrays[key].shape}")
        print(f"{key}_layout: {arrays[key + '_layout']}")
        print(f"{key}_faces_per_step: {arrays[key + '_faces_per_step']}")
        print(f"{key}_printed_shape: {arrays[key + '_printed_shape']}")
        print(f"{key}_full_periodic_shape: {arrays[key + '_full_periodic_shape']}")

    if args.out:
        np.savez_compressed(args.out, **arrays)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
