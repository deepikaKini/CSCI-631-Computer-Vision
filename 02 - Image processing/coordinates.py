import cv2 as cv
import numpy as np
import argparse
from pathlib import Path


###############################################
# Place your custom function definitions here #
def mouse_callback(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        print(f"Coordinates: x={x}, y={y}")
###############################################
def adjust_image(image: np.ndarray) -> np.ndarray:

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

    # return image
    return image


def main(args: argparse.Namespace) -> None:
    image = cv.imread(str(args.image))
    print(image)
    adjust_image(image)
    adjusted_image = adjust_image(image)
    save_name = args.image.stem + "_adjusted" + args.image.suffix
    cv.imwrite(str(save_name), adjusted_image)


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
