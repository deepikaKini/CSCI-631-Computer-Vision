import cv2 as cv
import numpy as np
from helpers import non_maximal_suppression
from pathlib import Path
from typing import Optional


class HoughCircleDetector:
    def __init__(self,
                 image_shape: tuple[int, int],
                 resolution: int,
                 radius: float,
                 soft_vote_sigma: Optional[float] = 8):
        h, w = image_shape

        # Create a grid of parameters (x and y centers of circles)
        self.center_x = np.linspace(0, w-1, resolution + 1)
        self.center_x = (self.center_x[:-1] + self.center_x[1:]) / 2
        self.center_y = np.linspace(0, h-1, resolution + 1)
        self.center_y = (self.center_y[:-1] + self.center_y[1:]) / 2

        self.radius = radius
        if soft_vote_sigma is None:
            # Sensible default for 'sigma' is the diagonal length of the image divided by the resolution
            self.sigma = np.sqrt(w**2 + h**2) / resolution
        else:
            self.sigma = soft_vote_sigma
        # self.sigma = 20
        # print(self.sigma)
        # Initialize self.accumulator to be a 2D array of zeros with the same shape as the parameter space. The value at
        # accumulator[i,j] represents the total number of "votes" for a circle centered at (center_y[i], center_x[j])
        self.accumulator = np.zeros(shape=(len(self.center_y), len(self.center_x)), dtype=float)

    def clear(self):
        self.accumulator = np.zeros(shape=(len(self.center_y), len(self.center_x)), dtype=float)

    def add_edge_at_xy(self, xy: tuple[float, float]):
        """
        Add an edge at the given (x, y) coordinate in image-space. Add a value to the accumulator for all circles
         that pass through this point, using a soft fall-off such that points that are "close" to the circle get a
         fraction of a vote.

         More precisely, the number of votes given to a circle with center at (cx, cy) is equal to

            np.exp(-dr**2 / self.sigma**2)

         where dr is the smallest distance between (x, y) and the circle centered at (cx, cy).
        """
        x, y  = xy

        dr = self.distance_to_circles( self.center_x, self.center_y,   x, y, self.radius )
        # print(dr.shape)
        # Calculate the soft votes using the given formula
        votes = np.exp(-dr ** 2 / (self.sigma ** 2))
        # print(votes.shape)
        # print(self.accumulator.shape)

        # Increment the accumulator array by the computed votes at the calculated indices
        self.accumulator += votes



    def get_circles(self, threshold: float, nms_window: int) -> np.ndarray:
        """Return a list of circles (cx, cy) which have a vote count above the threshold and are a local maximum in
        the accumulator space.

        :param threshold: minumum number of 'votes', as a fraction of the maximum number of votes
        :param nms_window: window size for non-maximal suppression
        :return: numpy array of shape (num_circles, 2) where each row is (x, y) coordinate of the center of the circle
        """

        # max_votes = np.max(self.accumulator)
        # accumulatorthreshold_votes = threshold * max_votes
        # potential_centers = np.argwhere(self.accumulator >= threshold_votes)
        # print(self.accumulator.max())
        votes = self.accumulator / self.accumulator.max()
        # print(np.max(votes))
        votes[votes > threshold] = 0
        votes = non_maximal_suppression(votes, window=nms_window)

        x,y = np.where(votes > 0)
        # Extract the (x, y) coordinates of the selected circle centers
        # circle_centers = np.array([(self.center_x[x], self.center_y[y]) for x, y in zip(x, y)])
        #
        # return circle_centers

        return np.stack([x, y], axis= -1)

    def distance_to_circles(self,centerx,centery, x, y, r):
        # Calculate the squared distances from xy to each center in centers_xy
        centers_x, centers_y = np.meshgrid(centerx, centery)
        squared_distances = ((centers_x - x)**2+(centers_y - y)**2)

        # Calculate the distances by taking the square root of the squared distances
        distances = np.sqrt(squared_distances) - r

        # Make sure distances are non-negative
        distances = np.abs(distances)
        # print(distances)

        return distances
def main(image: np.ndarray,
         canny_blur: int,
         canny_threshold_1: float,
         canny_threshold_2: float,
         accumulator_threshold: float,
         nms_window: int,
         resolution: int,
         radius: float) -> np.ndarray:
    annotated_image = image.copy()

    # Convert to grayscale.
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the image before running Canny edge detection.
    image = cv.GaussianBlur(image, (canny_blur, canny_blur), 0)

    # Run Canny edge detection.
    edges = cv.Canny(image, canny_threshold_1, canny_threshold_2)
    cv.imshow("canny", edges)
    cv.imwrite("output_images/aa.jpg", edges)
    cv.waitKey(0)
    # Create a HoughCircleDetector object.
    hough = HoughCircleDetector((image.shape[0], image.shape[1]), resolution, radius)

    # Iterate over the edges and add each edge to the HoughCircleDetector.
    for y, x in np.argwhere(edges > 0):
        hough.add_edge_at_xy((x, y))

    # Get the circles from the HoughCircleDetector.
    circles = hough.get_circles(accumulator_threshold, nms_window)
    # cv.imshow("circles", circles)
    # cv.waitKey(0)
    # Draw the circles on the original image.
    for cx, cy in circles:
        print(cx, cy)
        cv.circle(annotated_image, (int(hough.center_x[cx]), int(hough.center_y[cy])), int(radius + 0.5), (0, 0, 255), 1, cv.LINE_AA)
        
    # cv.circle(annotated_image, (500,72), int(radius + 0.5), (0, 0, 255), 4, cv.LINE_AA)

    return annotated_image


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image",
                        type=Path,
                        help="Path to image file")
    parser.add_argument("--accumulator-threshold",
                        type=float,
                        default=0.95,#0.9
                        help="Threshold for circle detection, relative to max # of votes")
    parser.add_argument("--nms-window",
                        type=int,
                        default=11,
                        help="Window size for non-maximal suppression")
    parser.add_argument("--resolution",
                        type=int,
                        default=100,
                        help="Number of parameters for each of center x and center y")
    parser.add_argument("--radius",
                        type=float,
                        default=48,
                        help="Radius of circles to detect")
    parser.add_argument("--canny-blur",
                        type=int,
                        default= 9,
                        help="Amount of Gaussian blur to apply before Canny edge detection")
    parser.add_argument("--canny-threshold-1",
                        type=float,
                        default=100,
                        help="Low threshold for Canny edge detection")
    parser.add_argument("--canny-threshold-2",
                        type=float,
                        default=200,
                        help="High threshold for Canny edge detection")
    args = parser.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Could not find image file: {args.image}")

    out_file = args.image.parent / "output_images" / f"{args.image.stem}_{int(args.radius)}.jpg"

    kwargs = vars(args)
    kwargs["image"] = cv.imread(str(args.image))
    image_with_circles = main(**kwargs)


    cv.imwrite(str(out_file), image_with_circles)
