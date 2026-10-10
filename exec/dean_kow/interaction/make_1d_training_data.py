#!/usr/bin/env python3
"""Create run_17-style 1D training samples from loaded current arrays.

Input is the .npz produced by load_current_ascii.py. The conversion treats the
2D data as independent 1D systems:
  x-current samples: fixed j lines, density[:, :, j], current_x[:, :, j]
  y-current samples: fixed i lines, density[:, i, :], current_y[:, i, :]
"""

from __future__ import annotations

import argparse

import numpy as np


def _normalize_directions(directions: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(directions, str):
        directions = (directions,)
    out: list[str] = []
    for direction in directions:
        if direction == "both":
            out.extend(["x", "y"])
        elif direction in {"x", "y"}:
            out.append(direction)
        else:
            raise ValueError("directions must contain only 'x', 'y', or 'both'")
    return tuple(dict.fromkeys(out))


def _periodic_face_flux(flux_line: np.ndarray, ncells: int) -> np.ndarray:
    """Return one flux value per unique periodic face."""
    nfaces = flux_line.shape[-1]
    if nfaces == ncells:
        return flux_line
    if nfaces == ncells + 1:
        return flux_line[..., :ncells]
    raise ValueError(f"expected {ncells} or {ncells + 1} faces, got {nfaces}")


def make_1d_training_data(
    arrays: dict[str, np.ndarray],
    *,
    half_window: int,
    history_length: int,
    directions: str | list[str] | tuple[str, ...] = ("x", "y"),
    density_scale: float = 1.0,
    periodic: bool = True,
    dtype: np.dtype | str = np.float32,
) -> dict[str, np.ndarray]:
    """Create face-local 1D samples matching run_17 tensor shapes.

    Returns:
      input_batch_N: (samples, history_length+1, 2*half_window)
      input_batch_F: (samples, history_length, 1)
      output_batch:  (samples, 1, 1)

    For a face index f, the density window is
    [f-half_window, ..., f+half_window-1] with periodic wrapping. This matches
    helpers_extended_domain.py, where the target face is between the two center
    cells of the 2*half_window density stencil.
    """
    if not periodic:
        raise NotImplementedError("only periodic compact 1D training data is supported")
    if half_window <= 0:
        raise ValueError("half_window must be positive")
    if history_length < 0:
        raise ValueError("history_length must be non-negative")
    if "density" not in arrays:
        raise KeyError("missing density array 'density'")

    dtype = np.dtype(dtype)
    density = np.asarray(arrays["density"], dtype=dtype) * dtype.type(density_scale)
    nstates, nx, ny = density.shape
    ntransitions = nstates - 1
    if history_length >= ntransitions:
        raise ValueError(
            "history_length must be smaller than the number of current "
            f"transitions ({history_length} >= {ntransitions})"
        )

    requested = _normalize_directions(directions)
    n_starts = ntransitions - history_length
    total_samples = 0
    for direction in requested:
        if direction == "x":
            total_samples += n_starts * ny * nx
        elif direction == "y":
            total_samples += n_starts * nx * ny

    input_batch_N = np.empty(
        (total_samples, history_length + 1, 2 * half_window), dtype=dtype
    )
    input_batch_F = np.empty((total_samples, history_length, 1), dtype=dtype)
    output_batch = np.empty((total_samples, 1, 1), dtype=dtype)
    sample_direction = np.empty(total_samples, dtype="U1")
    sample_line_index = np.empty(total_samples, dtype=np.int64)
    sample_face_index = np.empty(total_samples, dtype=np.int64)
    sample_time_index = np.empty(total_samples, dtype=np.int64)

    pos = 0
    for direction in requested:
        current_key = f"current_{direction}"
        if current_key not in arrays:
            raise KeyError(f"missing current array {current_key!r}")
        current = np.asarray(arrays[current_key], dtype=dtype)
        if current.shape[0] != ntransitions:
            raise ValueError(
                f"{current_key} must contain {ntransitions} transitions, "
                f"got {current.shape[0]}"
            )

        if direction == "x":
            printed_shape = arrays.get("current_x_printed_shape", current.shape[1:])
            nfaces = int(printed_shape[0])
            nlines = min(ny, int(printed_shape[1]))
            current_compact = current[:, :nfaces, :nlines]

            for line in range(nlines):
                density_line = density[:, :, line]
                flux_line = _periodic_face_flux(current_compact[:, :, line], nx)
                for start in range(n_starts):
                    density_hist = density_line[start : start + history_length + 1]
                    flux_hist = flux_line[start : start + history_length]
                    flux_out = flux_line[start + history_length]
                    for face in range(nx):
                        stencil = np.arange(face - half_window, face + half_window) % nx
                        input_batch_N[pos] = density_hist[:, stencil]
                        input_batch_F[pos, :, 0] = flux_hist[:, face]
                        output_batch[pos, 0, 0] = flux_out[face]
                        sample_direction[pos] = direction
                        sample_line_index[pos] = line
                        sample_face_index[pos] = face
                        sample_time_index[pos] = start
                        pos += 1

        elif direction == "y":
            printed_shape = arrays.get("current_y_printed_shape", current.shape[1:])
            nlines = min(nx, int(printed_shape[0]))
            nfaces = int(printed_shape[1])
            current_compact = current[:, :nlines, :nfaces]

            for line in range(nlines):
                density_line = density[:, line, :]
                flux_line = _periodic_face_flux(current_compact[:, line, :], ny)
                for start in range(n_starts):
                    density_hist = density_line[start : start + history_length + 1]
                    flux_hist = flux_line[start : start + history_length]
                    flux_out = flux_line[start + history_length]
                    for face in range(ny):
                        stencil = np.arange(face - half_window, face + half_window) % ny
                        input_batch_N[pos] = density_hist[:, stencil]
                        input_batch_F[pos, :, 0] = flux_hist[:, face]
                        output_batch[pos, 0, 0] = flux_out[face]
                        sample_direction[pos] = direction
                        sample_line_index[pos] = line
                        sample_face_index[pos] = face
                        sample_time_index[pos] = start
                        pos += 1

    return {
        "input_batch_N": input_batch_N[:pos],
        "input_batch_F": input_batch_F[:pos],
        "output_batch": output_batch[:pos],
        "sample_direction": sample_direction[:pos],
        "sample_line_index": sample_line_index[:pos],
        "sample_face_index": sample_face_index[:pos],
        "sample_time_index": sample_time_index[:pos],
        "half_window": np.array(half_window, dtype=np.int64),
        "history_length": np.array(history_length, dtype=np.int64),
        "density_scale": np.array(density_scale, dtype=float),
    }


def load_npz(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create 1D run_17-style training samples from loaded current arrays."
    )
    parser.add_argument("input_npz", help=".npz written by load_current_ascii.py --out")
    parser.add_argument("--out", required=True, help="output training .npz path")
    parser.add_argument("--half-window", type=int, required=True, help="density half-window")
    parser.add_argument(
        "--history-length",
        type=int,
        required=True,
        help="number of past current steps to include",
    )
    parser.add_argument(
        "--directions",
        nargs="+",
        choices=("x", "y", "both"),
        default=["both"],
        help="which 1D directions to generate",
    )
    parser.add_argument(
        "--density-scale",
        type=float,
        default=1.0,
        help="multiply density histories by this value before creating samples",
    )
    args = parser.parse_args()

    arrays = load_npz(args.input_npz)
    training = make_1d_training_data(
        arrays,
        half_window=args.half_window,
        history_length=args.history_length,
        directions=args.directions,
        density_scale=args.density_scale,
    )

    np.savez_compressed(args.out, **training)
    print(f"input_batch_N: {training['input_batch_N'].shape}")
    print(f"input_batch_F: {training['input_batch_F'].shape}")
    print(f"output_batch: {training['output_batch'].shape}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
