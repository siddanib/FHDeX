#!/usr/bin/env python3
"""Compute temporal Pearson correlations from face rows in ASCII .dat files."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_PATTERN = "plt_pp_*_current_ascii.dat"
TIMESTEP_RE = re.compile(r"(\d+)")


def timestep_from_name(path: Path) -> int:
    matches = TIMESTEP_RE.findall(path.stem)
    if not matches:
        raise ValueError(f"Could not find a timestep number in {path.name}")
    return int(matches[-1])


def read_face_values(path: Path) -> dict[tuple[int, str, int, int, int], float]:
    values: dict[tuple[int, str, int, int, int], float] = {}

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line or line.startswith("#"):
                continue

            fields = line.split()
            if not fields or fields[0] != "face":
                continue

            if len(fields) != 7:
                raise ValueError(
                    f"{path}:{line_number}: expected 7 fields for a face row, "
                    f"got {len(fields)}"
                )

            key = (
                int(fields[1]),
                fields[2],
                int(fields[3]),
                int(fields[4]),
                int(fields[5]),
            )
            values[key] = float(fields[6])

    if not values:
        raise ValueError(f"No face rows found in {path}")

    return values


def pearson(x: list[float], y: list[float]) -> float:
    if len(x) != len(y):
        raise ValueError("Pearson inputs must have the same length")
    if len(x) < 2:
        return math.nan

    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)

    numerator = 0.0
    sum_x2 = 0.0
    sum_y2 = 0.0
    for xi, yi in zip(x, y):
        dx = xi - mean_x
        dy = yi - mean_y
        numerator += dx * dy
        sum_x2 += dx * dx
        sum_y2 += dy * dy

    denominator = math.sqrt(sum_x2 * sum_y2)
    if denominator == 0.0:
        return math.nan
    return numerator / denominator


def load_snapshots(files: list[Path]) -> tuple[list[int], list[tuple[int, str, int, int, int]], list[list[float]]]:
    snapshots = [(timestep_from_name(path), path, read_face_values(path)) for path in files]
    snapshots.sort(key=lambda item: item[0])

    first_keys = set(snapshots[0][2])
    for _, path, values in snapshots[1:]:
        keys = set(values)
        if keys != first_keys:
            missing = first_keys - keys
            extra = keys - first_keys
            raise ValueError(
                f"Face keys in {path} do not match the first file "
                f"({len(missing)} missing, {len(extra)} extra)"
            )

    face_keys = sorted(first_keys)
    timesteps = [timestep for timestep, _, _ in snapshots]
    vectors = [[values[key] for key in face_keys] for _, _, values in snapshots]
    return timesteps, face_keys, vectors


def write_matrix(path: Path, timesteps: list[int], matrix: list[list[float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestep"] + timesteps)
        for timestep, row in zip(timesteps, matrix):
            writer.writerow([timestep] + row)


def write_reference(path: Path, timesteps: list[int], matrix: list[list[float]], reference_index: int) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestep", "pearson_r_to_reference"])
        for timestep, row in zip(timesteps, matrix):
            writer.writerow([timestep, row[reference_index]])


def correlations_by_time_difference(
    timesteps: list[int], matrix: list[list[float]]
) -> list[tuple[int, float, int]]:
    grouped: dict[int, list[float]] = {}
    for i, left_timestep in enumerate(timesteps):
        for j in range(i, len(timesteps)):
            delta_t = timesteps[j] - left_timestep
            value = matrix[i][j]
            if not math.isnan(value):
                grouped.setdefault(delta_t, []).append(value)

    return [
        (delta_t, sum(values) / len(values), len(values))
        for delta_t, values in sorted(grouped.items())
    ]


def write_time_difference(path: Path, rows: list[tuple[int, float, int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["delta_timestep", "mean_pearson_r", "pair_count"])
        writer.writerows(rows)


def plot_time_difference(path: Path, rows: list[tuple[int, float, int]]) -> None:
    delta_t = [row[0] for row in rows]
    mean_r = [row[1] for row in rows]

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(delta_t, mean_r, marker="o", linewidth=1.5, markersize=4)
    ax.axhline(0.0, color="0.4", linewidth=0.8)
    ax.set_xlabel("Difference in timestep")
    ax.set_ylabel("Mean Pearson r")
    ax.set_title("Temporal correlation of face integrated_current")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Pearson correlation between timestep snapshots using only "
            "face rows: face lev dir i j k integrated_current."
        )
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help=f"Glob pattern for input files, relative to --directory. Default: {DEFAULT_PATTERN}",
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path("."),
        help="Directory containing the .dat files. Default: current directory",
    )
    parser.add_argument(
        "--matrix-output",
        type=Path,
        default=Path("temporal_face_pearson_matrix.csv"),
        help="Output CSV for the full timestep-by-timestep correlation matrix.",
    )
    parser.add_argument(
        "--reference-output",
        type=Path,
        default=Path("temporal_face_pearson_to_first.csv"),
        help="Output CSV for correlation against the reference timestep.",
    )
    parser.add_argument(
        "--reference-timestep",
        type=int,
        default=None,
        help="Reference timestep for the compact output. Default: first timestep found.",
    )
    parser.add_argument(
        "--time-difference-output",
        type=Path,
        default=Path("temporal_face_pearson_by_delta_t.csv"),
        help="Output CSV for mean Pearson correlation by timestep difference.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=Path("temporal_face_pearson_by_delta_t.png"),
        help="Output image for mean Pearson correlation by timestep difference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(args.directory.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files matched {args.directory / args.pattern}")

    timesteps, face_keys, vectors = load_snapshots(files)

    if args.reference_timestep is None:
        reference_index = 0
    else:
        try:
            reference_index = timesteps.index(args.reference_timestep)
        except ValueError as exc:
            raise SystemExit(
                f"Reference timestep {args.reference_timestep} was not found. "
                f"Available range: {timesteps[0]} to {timesteps[-1]}"
            ) from exc

    matrix = [[pearson(left, right) for right in vectors] for left in vectors]
    by_time_difference = correlations_by_time_difference(timesteps, matrix)
    write_matrix(args.matrix_output, timesteps, matrix)
    write_reference(args.reference_output, timesteps, matrix, reference_index)
    write_time_difference(args.time_difference_output, by_time_difference)
    plot_time_difference(args.plot_output, by_time_difference)

    print(f"Read {len(files)} files with {len(face_keys)} aligned face rows each.")
    print(f"Wrote full correlation matrix: {args.matrix_output}")
    print(
        "Wrote reference correlations "
        f"(reference timestep {timesteps[reference_index]}): {args.reference_output}"
    )
    print(f"Wrote mean correlation by timestep difference: {args.time_difference_output}")
    print(f"Wrote plot: {args.plot_output}")


if __name__ == "__main__":
    main()
