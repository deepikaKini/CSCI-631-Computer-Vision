import cv2 as cv
import numpy as np
from helpers import non_maximal_suppression
from points_and_lines import get_all_lines_through_points
from pathlib import Path

"""
The numpy.linspace function is used to create an array of evenly spaced numbers over a specified range. It is often used to generate a sequence of values that are evenly distributed between a starting and ending point, with a specified number of elements.
"""
class HoughLineDetector:
    def __init__(self,
                 image_shape: tuple[int, int],
                 num_offsets: int,
                 num_angles: int):
        h, w = image_shape

        # We'll use the center of the image as our "origin" for the coordinate system for lines.
        self.origin_xy = (w / 2, h / 2)

        # Largest possible offset is the distance from the origin to the corner of the image.
        max_offset = np.sqrt(h ** 2 + w ** 2) / 2

        # Create a coordinate system of offsets (rho) and angles (theta) for the parameter space. Angles are in
        # radians, and we create 'num_angles+1' of them then remove the last one so that things remain equally spaced
        # (as opposed to duplicating -pi/2 and pi/2, which are effectively the same angle)
        self.offsets = np.linspace(-max_offset, max_offset, num_offsets)
        self.angles = np.linspace(-np.pi/2, np.pi/2, num_angles + 1)[:num_angles]

        # Initialize self.accumulator to be a 2D array of zeros with the same shape as the parameter space.
        self.accumulator = np.zeros(shape=(len(self.offsets), len(self.angles)), dtype=float)

    def clear(self):
        self.accumulator = np.zeros(shape=(len(self.offsets), len(self.angles)), dtype=float)

    def add_edge_at_xy(self, xy: tuple[float, float], horizontal_window_degrees):
        """Add an edge at the given (x, y) coordinate in image-space. Add a value to the accumulator for all lines that
        pass through this point.
        """
        dx, dy = xy[0] - self.origin_xy[0], xy[1] - self.origin_xy[1]
        rho_theta = get_all_lines_through_points(np.array([dx, dy]), self.angles)[0]
        rho_values_per_angle = rho_theta[:, 0]

        # Find the index in the 'offsets' array corresponding to each value of r_per_a.
        mask = np.logical_and(rho_values_per_angle >= self.offsets.min(), rho_values_per_angle < self.offsets.max())
        a_idx = np.arange(len(self.angles))[mask]
        r_fraction = (rho_values_per_angle - self.offsets.min()) / (self.offsets.max() - self.offsets.min())
        r_idx = (len(self.offsets) * r_fraction).astype(np.int32)

        # Increment the accumulator (vectorized over all angles)
        self.accumulator[r_idx, a_idx] = self.accumulator[r_idx, a_idx] + 1



    def get_lines(self, threshold: float, nms_window: int) -> np.ndarray:
        """Return a list of lines (rho, theta) which have a vote count above the threshold and are a local maximum
        in the accumulator space.

        :param threshold: minumum number of 'votes', as a fraction of the maximum number of votes
        :param nms_window: window size for non-maximal suppression
        :return: numpy array of shape (num_lines, 2) where each row is (offset, angle)
        """
        votes = self.accumulator / self.accumulator.max()
        print(np.max(votes))
        votes[votes < threshold] = 0
        votes = non_maximal_suppression(votes, window=nms_window)
        # cv.imshow("lines", votes)
        # cv.waitKey(0)
        ii, jj = np.where(votes > 0)
        return np.stack([self.offsets[ii], self.angles[jj]], axis=-1)

    def line_to_p(self, offset, angle):
        """Convert a line (rho, theta) to the point (x, y) in the image coordinate system on the line closest to the
        origin.
        """
        return np.array([np.cos(angle), np.sin(angle)]) * offset + np.array(self.origin_xy)

    def line_to_xy_endpoints(self, offset, angle):
        """Convert a line (offset, angle) to a pair of points (x1, y1), (x2, y2) which are the endpoints of the line.
        """
        length = float(self.offsets.max() - self.offsets.min())
        p = self.line_to_p(offset, angle)
        v = np.array([np.sin(angle), -np.cos(angle)])
        return p + v * length / 2, p - v * length / 2


def main(image: np.ndarray,
         canny_blur: int,
         canny_threshold_1: float,
         canny_threshold_2: float,
         accumulator_threshold: float,
         nms_window: int,
         num_offsets: int,
         num_angles: int,
         horizontal_window_degrees) -> np.ndarray:
    annotated_image = image.copy()

    # Convert to grayscale.
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow("gray", image)
    cv.waitKey(0)

    # Apply Gaussian blur to the image before running Canny edge detection.
    image = cv.GaussianBlur(image, (canny_blur, canny_blur), 0)
    cv.imshow("blur", image)
    cv.waitKey(0)
    # Run Canny edge detection.
    edges = cv.Canny(image, canny_threshold_1, canny_threshold_2)
    cv.imshow("canny", edges)
    cv.waitKey(0)

    # Create a HoughLineDetector object.
    hough = HoughLineDetector(image.shape[:2], num_offsets, num_angles)

    # Iterate over the edges and add each edge to the HoughLineDetector.
    for y, x in np.argwhere(edges > 0):
        hough.add_edge_at_xy((x, y),  horizontal_window_degrees)

    # Get the lines from the HoughLineDetector.
    lines = hough.get_lines(accumulator_threshold, nms_window)
    # cv.imshow("lines", lines)
    # cv.waitKey(0)
    print(lines)
    # Draw the lines on the original image.
    for offset, angle in lines:
        if np.degrees(angle) > (90 - horizontal_window_degrees) and np.degrees(angle) < (90 + horizontal_window_degrees ):
            print(angle)
            p1, p2 = hough.line_to_xy_endpoints(offset, angle)
            cv.line(annotated_image, tuple(p1.astype(int)), tuple(p2.astype(int)), (0, 0, 255), 1, cv.LINE_AA)

    return annotated_image


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image",
                        type=Path,
                        help="Path to image file")
    parser.add_argument("--accumulator-threshold",
                        type=float,
                        default=0.6,#default 0.6
                        help="Threshold for line detection, relative to max # of votes")
    parser.add_argument("--nms-window",
                        type=int,
                        default=7,
                        help="Window size for non-maximal suppression")
    parser.add_argument("--num-offsets",
                        type=int,
                        default=300,
                        help="Number of offsets to use in parameter space")
    parser.add_argument("--num-angles",
                        type=int,
                        default=360,
                        help="Number of angles to use in parameter space")
    parser.add_argument("--canny-blur",
                        type=int,
                        default=5,
                        help="Amount of Gaussian blur to apply before Canny edge detection")
    parser.add_argument("--canny-threshold-1",
                        type=float,
                        default=25,
                        help="Low threshold for Canny edge detection")
    parser.add_argument("--canny-threshold-2",
                        type=float,
                        default=150,
                        help="High threshold for Canny edge detection")
    parser.add_argument("--suffix",
                        type=str,
                        default="lines",
                        help="Suffix to append to output file name")
    # UNCOMMENT THIS AND IMPLEMENT CHANGES TO THIS FILE TO ENABLE THIS FEATURE
    parser.add_argument("--horizontal-window-degrees",
                        type=float,
                        default=5.0,
                        help="A line is defined as horizontal if it is within plus-or-minus this many degrees of horizontal")
    args = parser.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Could not find image file: {args.image}")

    out_file = args.image.parent / "output_images" / f"{args.image.stem}_{args.suffix}.jpg"

    kwargs = vars(args)
    kwargs["image"] = cv.imread(str(args.image))
    kwargs.pop("suffix")
    print(kwargs)
    image_with_lines = main(**kwargs)

    cv.imwrite(str(out_file), image_with_lines)
