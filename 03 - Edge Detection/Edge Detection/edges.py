import cv2 as cv
import numpy as np
import argparse
from pathlib import Path
from helpers import conv2D
from helpers import non_maximal_suppression


def my_edge_detect(image: np.ndarray, filter_x: np.ndarray, filter_y: np.ndarray):
    # Get gradients in x and y directions
    grad_x = conv2D(image, filter_x)
    grad_y = conv2D(image, filter_y)

    # Get magnitude and direction of edges
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    direction = np.arctan2(grad_y, grad_x)

    return magnitude, direction


#
# def non_maximal_suppression_dir(image, gradient_magnitude, gradient_direction):
#     height, width = image.shape
#     edge_image = np.zeros((height, width), dtype=np.uint8)
#
#     # Define quantized gradient directions (e.g., 0°, 45°, 90°, 135°)
#     quantized_directions = [0,  90]
#
#     for y in range(1, height - 1):
#         for x in range(1, width - 1):
#             angle = gradient_direction[y, x]
#
#             # Quantize the gradient direction to one of the predefined directions
#             quantized_angle = min(quantized_directions, key=lambda qd: abs(qd - angle))
#
#             # Determine neighbor pixels based on the quantized angle
#             if quantized_angle == 0:
#                 neighbors = [(y, x - 1), (y, x + 1)]
#             elif quantized_angle == 45:
#                 neighbors = [(y - 1, x - 1), (y + 1, x + 1)]
#             elif quantized_angle == 90:
#                 neighbors = [(y - 1, x), (y + 1, x)]
#             else:  # quantized_angle == 135
#                 neighbors = [(y - 1, x + 1), (y + 1, x - 1)]
#
#             # Compare the gradient magnitude of the current pixel with its neighbors
#             current_magnitude = gradient_magnitude[y, x]
#             neighbor_magnitudes = [gradient_magnitude[ny, nx] for ny, nx in neighbors]
#
#             # If the current pixel has the maximum magnitude, mark it as an edge point
#             if current_magnitude >= max(neighbor_magnitudes):
#                 edge_image[y, x] = 255  # Mark as an edge point
#
#     # Display the result
#     cv.imshow('Edge Image', edge_image)
#     cv.waitKey(0)
#
#     return edge_image

def edge_display(magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Given magnitude and direction of edges, create a color image to display them.

    The direction of the edges is encoded as the hue of the HSV image, and the
    magnitude is encoded as the value. The saturation is always 1.

    Assumes magnitude has nonnegative values and direction has values in radians.
    """
    magnitude = magnitude / magnitude.max()
    hue = direction % (2 * np.pi) / (2 * np.pi) * 180 / 255
    saturation = np.ones_like(magnitude)
    value = 0.1 + 0.9 * magnitude
    hsv_float = np.stack([hue, saturation, value], axis=-1)
    bgr = cv.cvtColor(np.clip(hsv_float * 255, 0, 255).astype("uint8"), cv.COLOR_HSV2BGR)
    return bgr


def main(image, kernel_size, gaussian_boolean, threshold, save_file=None):
    # Note that these filters are "flipped" from their usual definition because
    # we're using convolution instead of correlation.
    filter_x = np.array([[1, 0, -1],
                         [2, 0, -2],
                         [1, 0, -1]], dtype=np.float32)
    filter_y = np.array([[1, 2, 1],
                         [0, 0, 0],
                         [-1, -2, -1]], dtype=np.float32)

    # YOUR CODE HERE. You need to decide on whatever other pre-processing you want to
    # do either to the image (e.g. downsampling) or to the filter (e.g. adding
    # Gaussian blur). These values may change depending on the image you're using,
    # so you may want to make them command-line arguments and parameters to this
    # function. The helpers.filter_filter function may be useful.
    ...

    # ============
    # apply guassian blur
    # not required for van gogh
    if gaussian_boolean:
        print(gaussian_boolean)
        gaussian_kernel_1d = cv.getGaussianKernel(kernel_size, -1)
        gaussian_kernel_1d = gaussian_kernel_1d / np.sum(gaussian_kernel_1d)
        gaussian_kernel_2d = gaussian_kernel_1d @ gaussian_kernel_1d.T

        image = cv.filter2D(image, -1, gaussian_kernel_2d, borderType=cv.BORDER_REPLICATE)
        cv.imshow("gaussian applied", image)
        cv.waitKey(0)

    # ============

    # Run the edge detector to get magnitude and direction of edges in the image
    mag, dir = my_edge_detect(image, filter_x, filter_y)

    # YOUR CODE HERE. Do you want to do any post-processing on the magnitude and
    # direction? For example, if you downsampled the original image, you could
    # upsample mag and dir to match the original size. You could add a 'threshold'
    # parameter and reject any magnitude values (set them to zero) if they are below
    # the threshold. What does non-maximal suppression on the magnitude do to the
    # result? Explore some options to get the best edge images you can.
    ...
    # ============
    # 1.2 good for van gogh

    ##thresholding
    print(mag.shape)
    print(mag.max())
    mask1 = (mag < threshold)
    mag[mask1] = 0
    mag[mag > threshold] = 255.0

    # Apply non-maximal suppression
    # edge_image = non_maximal_suppression_dir(image, mag, dir)

    # Convert the magnitude and direction to a color image where magnitude is brightness
    # and direction is hue.
    edge_image = edge_display(mag, dir)


    # edge_image = non_maximal_suppression(edge_image,window =  11)
    # edge_image = cv.cvtColor(edge_image, cv.COLOR_BGR2GRAY)
    # print(edge_image.shape)
    # coordinates = non_maximal_suppression(edge_image)

    # =============
    # If a save file is given, write the output there. Otherwise, display it on screen.
    if save_file is not None:
        cv.imwrite(str(save_file), edge_image)
    else:
        cv.imshow("Edges", edge_image)
        cv.waitKey(0)
        cv.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--algorithm", default="sobel", choices=["sobel", "dog"])
    parser.add_argument("--gaussianBlur", default='0', choices=['0', '1'])
    parser.add_argument("--kernelSize", default= 11 )
    parser.add_argument("--threshold", default=0.5)
    ...
    args = parser.parse_args()

    if not args.image.exists():
        print(f"Image file {args.image} does not exist.")
        exit(1)

    save_file = args.image.parent / f"{args.image.stem}-out.jpg"
    image = cv.imread(str(args.image), cv.IMREAD_GRAYSCALE)

    # 11 kernel size, 0.50 threshold for brick (max mag is 1.07) --> pic 03
    # 13 0.2 (max mag is 0.8) for people -->  pic 02

    # ANY COMMAND LINE ARGUMENTS YOU ADD SHOULD BE PASSED TO THE MAIN() FUNCTION
    main(image, int(args.kernelSize), int(args.gaussianBlur), float(args.threshold), save_file=save_file)
