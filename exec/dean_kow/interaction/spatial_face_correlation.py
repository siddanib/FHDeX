#!/usr/bin/env python3
"""Compute spatial Pearson correlations from face integrated-current rows."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_PATTERN = "plt_pp_*_current_ascii.dat"
TIMESTEP_RE = re.compile(r"(\d+)")
FaceKey = tuple[int, str, int, int, int]


def timestep_from_name(path: Path) -> int:
    matches = TIMESTEP_RE.findall(path.stem)
    if not matches:
        raise ValueError(f"Could not find a timestep number in {path.name}")
    return int(matches[-1])


def read_face_values(path: Path) -> dict[FaceKey, float]:
    values: dict[FaceKey, float] = {}

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


def load_face_series(files: list[Path]) -> tuple[list[int], dict[FaceKey, list[float]]]:
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

    face_series = {key: [] for key in sorted(first_keys)}
    for _, _, values in snapshots:
        for key in face_series:
            face_series[key].append(values[key])

    timesteps = [timestep for timestep, _, _ in snapshots]
    return timesteps, face_series


def selected_face_series(
    face_series: dict[FaceKey, list[float]], face_dirs: set[str] | None
) -> dict[FaceKey, list[float]]:
    if face_dirs is None:
        return face_series
    return {key: series for key, series in face_series.items() if key[1] in face_dirs}


def pair_group_key(face_key: FaceKey, axis: str) -> tuple[int, str, int, int]:
    lev, face_dir, i, j, k = face_key
    if axis == "x":
        return lev, face_dir, j, k
    if axis == "y":
        return lev, face_dir, i, k
    raise ValueError(f"Unsupported axis {axis!r}")


def axis_index(face_key: FaceKey, axis: str) -> int:
    if axis == "x":
        return face_key[2]
    if axis == "y":
        return face_key[3]
    raise ValueError(f"Unsupported axis {axis!r}")


def fixed_labels(axis: str) -> tuple[str, str]:
    if axis == "x":
        return "j", "k"
    if axis == "y":
        return "i", "k"
    raise ValueError(f"Unsupported axis {axis!r}")


def spatial_correlations(
    face_series: dict[FaceKey, list[float]], axis: str, max_separation: int | None
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int]]]:
    grouped_faces: dict[tuple[int, str, int, int], list[FaceKey]] = {}
    for key in face_series:
        grouped_faces.setdefault(pair_group_key(key, axis), []).append(key)

    pair_rows: list[dict[str, float | int | str]] = []
    by_delta: dict[int, list[float]] = {}
    nan_by_delta: dict[int, int] = {}

    fixed_a_label, fixed_b_label = fixed_labels(axis)
    for group_key, keys in sorted(grouped_faces.items()):
        lev, face_dir, fixed_a, fixed_b = group_key
        keys = sorted(keys, key=lambda key: axis_index(key, axis))

        for left_index, left_key in enumerate(keys):
            left_position = axis_index(left_key, axis)
            for right_key in keys[left_index + 1 :]:
                right_position = axis_index(right_key, axis)
                delta = right_position - left_position
                if max_separation is not None and delta > max_separation:
                    break

                value = pearson(face_series[left_key], face_series[right_key])
                pair_rows.append(
                    {
                        "axis": axis,
                        "delta_index": delta,
                        "lev": lev,
                        "face_dir": face_dir,
                        fixed_a_label: fixed_a,
                        fixed_b_label: fixed_b,
                        f"left_{axis}": left_position,
                        f"right_{axis}": right_position,
                        "pearson_r": value,
                    }
                )

                if math.isnan(value):
                    nan_by_delta[delta] = nan_by_delta.get(delta, 0) + 1
                else:
                    by_delta.setdefault(delta, []).append(value)

    summary_rows: list[dict[str, float | int]] = []
    for delta in sorted(set(by_delta) | set(nan_by_delta)):
        values = by_delta.get(delta, [])
        nan_count = nan_by_delta.get(delta, 0)
        mean_r = sum(values) / len(values) if values else math.nan
        summary_rows.append(
            {
                "delta_index": delta,
                "mean_pearson_r": mean_r,
                "valid_pair_count": len(values),
                "nan_pair_count": nan_count,
                "total_pair_count": len(values) + nan_count,
            }
        )

    return pair_rows, summary_rows


def face_center(face_key: FaceKey) -> tuple[float, float]:
    _, face_dir, i, j, _ = face_key
    if face_dir == "x":
        return float(i), float(j) + 0.5
    if face_dir == "y":
        return float(i) + 0.5, float(j)
    return float(i), float(j)


def nearest_xy_correlations(
    face_series: dict[FaceKey, list[float]]
) -> tuple[list[dict[str, float | int]], list[dict[str, float | int]]]:
    x_faces = [key for key in face_series if key[1] == "x"]
    y_faces_by_plane: dict[tuple[int, int], list[FaceKey]] = {}
    for key in face_series:
        lev, face_dir, _, _, k = key
        if face_dir == "y":
            y_faces_by_plane.setdefault((lev, k), []).append(key)

    pair_rows: list[dict[str, float | int]] = []
    by_distance: dict[float, list[float]] = {}
    nan_by_distance: dict[float, int] = {}

    for x_key in sorted(x_faces):
        lev, _, x_i, x_j, k = x_key
        candidates = y_faces_by_plane.get((lev, k), [])
        if not candidates:
            continue

        x_center_i, x_center_j = face_center(x_key)
        distances = []
        for y_key in candidates:
            y_center_i, y_center_j = face_center(y_key)
            distance2 = (y_center_i - x_center_i) ** 2 + (y_center_j - x_center_j) ** 2
            distances.append((distance2, y_key))

        min_distance2 = min(distance2 for distance2, _ in distances)
        nearest = [y_key for distance2, y_key in distances if distance2 == min_distance2]

        for y_key in sorted(nearest):
            _, _, y_i, y_j, _ = y_key
            value = pearson(face_series[x_key], face_series[y_key])
            pair_rows.append(
                {
                    "lev": lev,
                    "k": k,
                    "x_i": x_i,
                    "x_j": x_j,
                    "y_i": y_i,
                    "y_j": y_j,
                    "center_distance": math.sqrt(min_distance2),
                    "pearson_r": value,
                }
            )

            if math.isnan(value):
                nan_by_distance[min_distance2] = nan_by_distance.get(min_distance2, 0) + 1
            else:
                by_distance.setdefault(min_distance2, []).append(value)

    summary_rows: list[dict[str, float | int]] = []
    for distance2 in sorted(set(by_distance) | set(nan_by_distance)):
        values = by_distance.get(distance2, [])
        nan_count = nan_by_distance.get(distance2, 0)
        mean_r = sum(values) / len(values) if values else math.nan
        summary_rows.append(
            {
                "center_distance": math.sqrt(distance2),
                "mean_pearson_r": mean_r,
                "valid_pair_count": len(values),
                "nan_pair_count": nan_count,
                "total_pair_count": len(values) + nan_count,
            }
        )

    return pair_rows, summary_rows


def write_rows(path: Path, rows: list[dict[str, float | int | str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_summaries(
    path: Path,
    x_rows: list[dict[str, float | int]],
    y_rows: list[dict[str, float | int]],
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    for axis, rows in (("x", x_rows), ("y", y_rows)):
        delta = [int(row["delta_index"]) for row in rows]
        mean_r = [float(row["mean_pearson_r"]) for row in rows]
        ax.plot(delta, mean_r, marker="o", linewidth=1.5, markersize=4, label=f"{axis} direction")

    ax.axhline(0.0, color="0.4", linewidth=0.8)
    ax.set_xlabel("Face separation in index units")
    ax.set_ylabel("Mean Pearson r")
    ax.set_title("Spatial correlation of face integrated_current")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute spatial Pearson correlations between face integrated-current "
            "time series. X correlations compare faces with matching lev, dir, j, k "
            "and different i. Y correlations compare faces with matching lev, dir, i, k "
            "and different j."
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
        "--face-dir",
        action="append",
        default=None,
        help=(
            "Only include this face orientation, for example --face-dir x. "
            "Can be repeated. Default: include all face dirs."
        ),
    )
    parser.add_argument(
        "--max-separation",
        type=int,
        default=None,
        help="Largest index separation to include. Default: include all separations.",
    )
    parser.add_argument(
        "--x-output",
        type=Path,
        default=Path("spatial_face_pearson_by_x.csv"),
        help="Output CSV for mean correlation by x-index separation.",
    )
    parser.add_argument(
        "--y-output",
        type=Path,
        default=Path("spatial_face_pearson_by_y.csv"),
        help="Output CSV for mean correlation by y-index separation.",
    )
    parser.add_argument(
        "--pair-output",
        type=Path,
        default=Path("spatial_face_pearson_pairs.csv"),
        help="Output CSV for every face-pair correlation. Use --no-pair-output to skip.",
    )
    parser.add_argument(
        "--no-pair-output",
        action="store_true",
        help="Do not write the pair-level correlation CSV.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=Path("spatial_face_pearson.png"),
        help="Output image comparing x and y spatial correlations.",
    )
    parser.add_argument(
        "--xy-nearest-output",
        type=Path,
        default=Path("spatial_face_pearson_x_to_nearest_y.csv"),
        help="Output CSV summarizing x-face correlations with nearest y-oriented faces.",
    )
    parser.add_argument(
        "--xy-nearest-pair-output",
        type=Path,
        default=Path("spatial_face_pearson_x_to_nearest_y_pairs.csv"),
        help="Output CSV for each x-face to nearest-y-face correlation pair.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(args.directory.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files matched {args.directory / args.pattern}")
    if args.max_separation is not None and args.max_separation < 1:
        raise SystemExit("--max-separation must be at least 1")

    timesteps, face_series = load_face_series(files)
    face_dirs = set(args.face_dir) if args.face_dir else None
    face_series = selected_face_series(face_series, face_dirs)
    if not face_series:
        raise SystemExit("No face rows matched the requested --face-dir filter")

    x_pairs, x_summary = spatial_correlations(face_series, "x", args.max_separation)
    y_pairs, y_summary = spatial_correlations(face_series, "y", args.max_separation)
    xy_nearest_pairs, xy_nearest_summary = nearest_xy_correlations(face_series)

    summary_fields = [
        "delta_index",
        "mean_pearson_r",
        "valid_pair_count",
        "nan_pair_count",
        "total_pair_count",
    ]
    write_rows(args.x_output, x_summary, summary_fields)
    write_rows(args.y_output, y_summary, summary_fields)
    write_rows(
        args.xy_nearest_output,
        xy_nearest_summary,
        [
            "center_distance",
            "mean_pearson_r",
            "valid_pair_count",
            "nan_pair_count",
            "total_pair_count",
        ],
    )
    write_rows(
        args.xy_nearest_pair_output,
        xy_nearest_pairs,
        ["lev", "k", "x_i", "x_j", "y_i", "y_j", "center_distance", "pearson_r"],
    )

    if not args.no_pair_output:
        pair_fields = [
            "axis",
            "delta_index",
            "lev",
            "face_dir",
            "i",
            "j",
            "k",
            "left_x",
            "right_x",
            "left_y",
            "right_y",
            "pearson_r",
        ]
        pair_rows = x_pairs + y_pairs
        write_rows(args.pair_output, pair_rows, pair_fields)

    plot_summaries(args.plot_output, x_summary, y_summary)

    print(
        f"Read {len(files)} files spanning timesteps {timesteps[0]} to {timesteps[-1]} "
        f"with {len(face_series)} face time series."
    )
    print(f"Wrote x-direction spatial correlation: {args.x_output}")
    print(f"Wrote y-direction spatial correlation: {args.y_output}")
    print(f"Wrote x-to-nearest-y spatial correlation: {args.xy_nearest_output}")
    print(f"Wrote x-to-nearest-y pair correlations: {args.xy_nearest_pair_output}")
    if not args.no_pair_output:
        print(f"Wrote pair-level spatial correlations: {args.pair_output}")
    print(f"Wrote plot: {args.plot_output}")


if __name__ == "__main__":
    main()
