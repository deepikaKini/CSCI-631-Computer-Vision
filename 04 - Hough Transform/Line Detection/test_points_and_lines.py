import unittest
import numpy as np
import points_and_lines


class TestGeometryFunctions(unittest.TestCase):
    def test_ensure_shape(self):
        self.assertTrue(
            np.array_equal(
                points_and_lines._ensure_shape(np.array([1, 2])), np.array([[1, 2]])
            )
        )
        self.assertTrue(
            np.array_equal(
                points_and_lines._ensure_shape(np.array([[1, 2], [3, 4]])),
                np.array([[1, 2], [3, 4]]),
            )
        )
        with self.assertRaises(ValueError):
            points_and_lines._ensure_shape(np.ones((2, 2, 2)))

    def test_cart2pol(self):
        np.testing.assert_almost_equal(
            points_and_lines.cart2pol(np.array([[1, 0], [0, 1], [-1, -1]])),
            np.array([[1, 0], [1, np.pi / 2], [np.sqrt(2), -3 * np.pi / 4]]),
            decimal=6,
        )

    def test_pol2cart(self):
        np.testing.assert_almost_equal(
            points_and_lines.pol2cart(
                np.array([[1, 0], [1, np.pi / 2], [np.sqrt(2), -3 * np.pi / 4]])
            ),
            np.array([[1, 0], [0, 1], [-1, -1]]),
            decimal=6,
        )

    def test_unit_vector(self):
        np.testing.assert_almost_equal(
            points_and_lines.unit_vector(np.array([[1, 0], [0, 1], [-1, -1]])),
            np.array([[1, 0], [0, 1], [-1 / np.sqrt(2), -1 / np.sqrt(2)]]),
            decimal=6,
        )

    def test_point_on_line_closest_to_origin(self):
        np.testing.assert_almost_equal(
            points_and_lines.point_on_line_closest_to_origin(
                np.array([[-1, -1], [1, 1], [1, 3], [3, -1], [6, -7]]),
                np.array([[1, 1], [-1, -1], [4, -3], [4, -3], [4, -3]]),
            ),
            np.array([[0, 0], [0, 0], [2, 1], [2, 1], [2, 1]]),
            decimal=6,
        )

    def test_distance_point_to_normal_line(self):
        np.testing.assert_almost_equal(
            points_and_lines.distance_point_to_normal_line(
                np.array(
                    [
                        [1, 0],
                        [1, 0],
                        [1, 0],
                        [1, 0],
                        [1, np.pi / 2],
                        [np.sqrt(2), np.pi / 4],
                        [np.sqrt(2), np.pi / 4],
                        [np.sqrt(2), np.pi / 4],
                    ]
                ),
                np.array(
                    [
                        [1, 0],
                        [1, 10],
                        [2, 10],
                        [0.5, 10],
                        [0, 1],
                        [1, 1],
                        [1, 0],
                        [2, 1],
                    ]
                ),
            ),
            np.array([0, 0, 1, 0.5, 0, 0, np.sqrt(2) / 2, np.sqrt(2) / 2]),
            decimal=6,
        )

    def test_get_all_lines_through_points(self):
        lines = points_and_lines.get_all_lines_through_points(
            np.array([[1, 0], [0, 1], [-1, -1]]),
            np.array([-np.pi, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2, np.pi]),
        )
        np.testing.assert_almost_equal(
            lines,
            np.array(
                [
                    [
                        [-1, -np.pi],
                        [0, -np.pi / 2],
                        [1 / np.sqrt(2), -np.pi / 4],
                        [1, 0],
                        [1 / np.sqrt(2), np.pi / 4],
                        [0, np.pi / 2],
                        [-1, np.pi],
                    ],
                    [
                        [0, -np.pi],
                        [-1, -np.pi / 2],
                        [-1 / np.sqrt(2), -np.pi / 4],
                        [0, 0],
                        [1 / np.sqrt(2), np.pi / 4],
                        [1, np.pi / 2],
                        [0, np.pi],
                    ],
                    [
                        [1, -np.pi],
                        [1, -np.pi / 2],
                        [0, -np.pi / 4],
                        [-1, 0],
                        [-np.sqrt(2), np.pi / 4],
                        [-1, np.pi / 2],
                        [1, np.pi],
                    ],
                ]
            ),
            decimal=6,
        )


if __name__ == "__main__":
    unittest.main()
