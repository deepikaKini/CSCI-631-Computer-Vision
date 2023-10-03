import cv2 as cv
import numpy as np
import argparse
from pathlib import Path

x_global = -1
y_global = -1

###############################################
def mouse_callback(event, x, y, flags, param):
    global  x_global, y_global
    if event == cv.EVENT_LBUTTONDOWN:
        x_global, y_global = x, y
        print(x_global)
###############################################
def crop_around_clicked(image: np.ndarray) -> np.ndarray:

    # Create a window for displaying the image
    cv.namedWindow('part 1 of 9')

    # Set the mouse callback function
    cv.setMouseCallback('part 1 of 9', mouse_callback)

    # Display the image
    cv.imshow('part 1 of 9', image)

    while True:

        k = cv.waitKey(1)
        if k == ord('q'):
            break

    # Close all windows
    cv.destroyAllWindows()
    #crop using coordinates
    print(x_global, y_global)
    im = image[x_global - 50:x_global + 50, y_global - 50:y_global + 50]
    return im


def scale_by_half_using_numpy_slicing(image: np.ndarray) -> np.ndarray:
    im = image[::2, ::2]
    return im

def scale_by_half_using_cv_resize(image: np.ndarray) -> np.ndarray:
    height, width, _ = image.shape

    new_height = height // 2
    new_width = width // 2
    im = cv.resize(image, (new_width, new_height))
    return im

def horizontal_mirror_image(image: np.ndarray) -> np.ndarray:
    #flip rows
    im = image[::-1]
    return im

def rotate_counterclockwise_90(image: np.ndarray) -> np.ndarray:
    # Rotate the image 90 degrees counterclockwise using np.transpose
    #counterclockwise (flipping x and y axis and not the BGR one(2))
    im = np.transpose(image, (1, 0, 2))
    #flip over vertical axis if clockwise required
    # im = im[:, ::-1, :]
    return im

def swap_b_r(image: np.ndarray) -> np.ndarray:
    # Split the channels
    b, g, r = cv.split(image)

    # Swap the Blue and Red channels
    im = cv.merge((r, g, b))
    return im

def invert_hue_ab(image: np.ndarray) -> np.ndarray:
    # Convert the image to LAB color space
    lab_image = cv.cvtColor(image, cv.COLOR_BGR2LAB)

    # Split the LAB image into channels
    l, a, b = cv.split(lab_image)

    # Invert the "a" channel and the "b" channel
    inverted_a = 255 - a
    inverted_b = 255 - b

    # Merge the inverted channels back into the LAB image
    inverted_lab_image = cv.merge((l, inverted_a, inverted_b))

    # # Convert the inverted LAB image back to BGR color space
    inverted_bgr_image = cv.cvtColor(inverted_lab_image, cv.COLOR_LAB2BGR)

    return inverted_bgr_image


def main(args: argparse.Namespace) -> None:
    image = cv.imread(str(args.image))
    image1  = crop_around_clicked(image)
    cv.imwrite("clicked.jpg", image1)

    image2 = scale_by_half_using_numpy_slicing(image)
    cv.imwrite("halfsize.jpg", image2)

    image3 = scale_by_half_using_cv_resize(image)
    cv.imwrite("halfsize_cv.jpg", image3)

    image4 = horizontal_mirror_image(image)
    cv.imwrite("flipped.jpg", image4)

    image5 = rotate_counterclockwise_90(image)
    cv.imwrite("rotated.jpg", image5)

    image6 = swap_b_r(image)
    cv.imwrite("swapped.jpg", image6)

    image7 = invert_hue_ab(image)
    cv.imwrite("inverted_ab.jpg", image7)

# Code inside the if statement below will only be executed if the script is called
# directly from the command line
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    args = parser.parse_args()

    if not args.image.exists():
        print(f"File {args.image} not found")
        exit(1)

    main(args)
