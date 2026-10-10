import tempfile
import unittest
from pathlib import Path

import numpy as np

from load_current_ascii import check_periodic_evolution, load_current_ascii


class LoadCurrentAsciiTests(unittest.TestCase):
    def write_file(self, directory: str, name: str, contents: str) -> Path:
        path = Path(directory, name)
        path.write_text(contents, encoding="utf-8")
        return path

    def test_accepts_blank_lines_and_leading_whitespace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_file(
                directory,
                "current_0007.dat",
                "\n  # comment\n  cell 2 4 8 0 1.5 2.5\n\n",
            )

            arrays = load_current_ascii(str(path), total_particle_count=1)

            self.assertEqual(arrays["level"], 2)
            np.testing.assert_array_equal(arrays["steps"], [7])
            np.testing.assert_array_equal(arrays["density_steps"], [6, 7])
            np.testing.assert_array_equal(arrays["density"][:, 0, 0], [1.5, 2.5])
            self.assertNotIn("density_old", arrays)
            self.assertNotIn("density_new", arrays)

    def test_reports_malformed_rows_with_file_and_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_file(directory, "current_1.dat", "cell 0 0 0 0 1\n")

            with self.assertRaisesRegex(
                ValueError, rf"{path}:1: expected 7 fields for a cell row, got 6"
            ):
                load_current_ascii(str(path))

    def test_rejects_multiple_k_planes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_file(
                directory,
                "current_1.dat",
                "cell 0 0 0 0 1 2\ncell 0 0 0 1 3 4\n",
            )

            with self.assertRaisesRegex(ValueError, "multiple k planes"):
                load_current_ascii(str(path))

    def test_rejects_duplicate_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_file(
                directory,
                "current_1.dat",
                "cell 0 0 0 0 1 2\ncell 0 0 0 0 3 4\n",
            )

            with self.assertRaisesRegex(ValueError, "duplicate cell coordinate"):
                load_current_ascii(str(path))

    def test_sparse_cells_are_nan_instead_of_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_file(
                directory,
                "current_1.dat",
                "cell 0 0 0 0 1 2\ncell 0 1 1 0 3 4\n",
            )

            arrays = load_current_ascii(str(path), total_particle_count=4)

            self.assertTrue(np.isnan(arrays["density"][1, 0, 1]))
            self.assertTrue(np.isnan(arrays["density"][1, 1, 0]))

    def test_padded_currents_use_directional_shapes_and_periodic_copies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_file(
                directory,
                "current_3.dat",
                "\n".join(
                    [
                        "cell 0 0 0 0 1 2",
                        "cell 0 0 1 0 1 2",
                        "cell 0 1 0 0 1 2",
                        "cell 0 1 1 0 1 2",
                        "face 0 x 0 0 0 10",
                        "face 0 x 0 1 0 11",
                        "face 0 x 1 0 0 12",
                        "face 0 x 1 1 0 13",
                        "face 0 y 0 0 0 20",
                        "face 0 y 0 1 0 21",
                        "face 0 y 1 0 0 22",
                        "face 0 y 1 1 0 23",
                    ]
                )
                + "\n",
            )

            arrays = load_current_ascii(
                str(path), current_grid="padded", total_particle_count=4
            )

            self.assertEqual(arrays["current_x"].shape, (1, 3, 2))
            self.assertEqual(arrays["current_y"].shape, (1, 2, 3))
            np.testing.assert_array_equal(
                arrays["current_x"][0, 2, :], arrays["current_x"][0, 0, :]
            )
            np.testing.assert_array_equal(
                arrays["current_y"][0, :, 2], arrays["current_y"][0, :, 0]
            )
            np.testing.assert_array_equal(arrays["current_x_full_periodic_shape"], [3, 2])
            np.testing.assert_array_equal(arrays["current_y_full_periodic_shape"], [2, 3])

    def test_reads_total_particle_count_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_file(
                directory,
                "current_1.dat",
                "# total_particle_count: 20480\ncell 1 0 0 0 1 2\n",
            )

            arrays = load_current_ascii(str(path), level=1)

            self.assertEqual(arrays["total_particle_count"], 20480.0)
            self.assertEqual(
                arrays["average_particle_count_per_cell"], 20480.0
            )
            np.testing.assert_array_equal(
                arrays["density"][:, 0, 0], [20480.0, 40960.0]
            )

    def test_accepts_particle_count_override_for_legacy_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_file(
                directory, "current_1.dat", "cell 1 0 0 0 1 2\n"
            )

            arrays = load_current_ascii(
                str(path), level=1, total_particle_count=12345
            )

            self.assertEqual(arrays["total_particle_count"], 12345.0)

    def test_consolidates_consecutive_old_and_new_states(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = self.write_file(
                directory, "current_10.dat", "cell 1 0 0 0 1 2\n"
            )
            second = self.write_file(
                directory, "current_11.dat", "cell 1 0 0 0 2 3\n"
            )

            arrays = load_current_ascii(
                [str(first), str(second)],
                level=1,
                total_particle_count=2,
            )

            np.testing.assert_array_equal(arrays["density_steps"], [9, 10, 11])
            np.testing.assert_array_equal(
                arrays["density"][:, 0, 0], [2.0, 4.0, 6.0]
            )

    def test_checks_periodic_evolution_between_consecutive_steps(self) -> None:
        arrays = {
            "steps": np.array([10, 11]),
            "density_steps": np.array([9, 10, 11]),
            "density": np.array(
                [[[5.0], [5.0]], [[5.0], [5.0]], [[4.0], [6.0]]]
            ),
            "current_x": np.array([[[0.0], [0.0]], [[0.0], [1.0]]]),
            "current_y": np.zeros((2, 2, 1)),
            "origin": np.array([0, 0]),
            "current_x_origin": np.array([0, 0]),
            "current_y_origin": np.array([0, 0]),
            "total_particle_count": np.array(10.0),
        }

        report = check_periodic_evolution(arrays)

        np.testing.assert_array_equal(report["from_steps"], [9, 10])
        np.testing.assert_array_equal(report["to_steps"], [10, 11])
        np.testing.assert_array_equal(report["passed"], [True, True])
        self.assertLess(np.max(report["max_abs_continuity_residual"]), 1.0e-12)

    def test_periodic_evolution_check_detects_reversed_current(self) -> None:
        arrays = {
            "steps": np.array([10, 11]),
            "density_steps": np.array([9, 10, 11]),
            "density": np.array(
                [[[5.0], [5.0]], [[5.0], [5.0]], [[4.0], [6.0]]]
            ),
            "current_x": np.array([[[0.0], [0.0]], [[0.0], [-1.0]]]),
            "current_y": np.zeros((2, 2, 1)),
            "origin": np.array([0, 0]),
            "current_x_origin": np.array([0, 0]),
            "current_y_origin": np.array([0, 0]),
            "total_particle_count": np.array(10.0),
        }

        report = check_periodic_evolution(arrays)

        np.testing.assert_array_equal(report["passed"], [True, False])
        self.assertGreater(report["max_abs_continuity_residual"][1], 1.0)


if __name__ == "__main__":
    unittest.main()
