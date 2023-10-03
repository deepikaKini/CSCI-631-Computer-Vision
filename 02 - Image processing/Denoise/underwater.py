import cv2 as cv
import numpy as np
import argparse
from pathlib import Path

#tried removing guassian noise, but no change: means no guassian noise
###############################################
# Place your custom function definitions here #
###############################################
def adjust_image(image: np.ndarray) -> np.ndarray:
    # Define the original gamma value (as per cvaa book gamma is 2.2)
    original_gamma = 2.2  # Adjust this value based on the original correction

    # Apply inverse gamma correction
    inverse_gamma_corrected = np.power(image / 255.0, 1 / original_gamma) * 255.0

    # Convert the inverse gamma-corrected image back to uint8 format
    image = np.uint8(inverse_gamma_corrected)

    #Scale the red channel by 120% and the blue channel by 80%
    image[:, :, 0] = np.clip(image[:, :, 0] * 1.2, 0, 255)
    image[:, :, 2] = np.clip(image[:, :, 2] * 0.8, 0, 255)

    # Convert from BGR to LAB
    lab_image = cv.cvtColor(image, cv.COLOR_BGR2LAB)

    # Split the LAB image into channels
    l, a, b = cv.split(lab_image)

    # Equalize the histogram of the L channel for improved contrast
    l_equalized = cv.equalizeHist(l)

    # Merge the equalized L channel with the original A and B channels
    lab_equalized = cv.merge((l_equalized, a, b))

    # Convert the LAB image back to BGR color space
    im = cv.cvtColor(lab_equalized, cv.COLOR_LAB2BGR)

    # Apply gamma correction
    gamma_corrected = np.power(im / 255.0, 1 / original_gamma) * 255.0

    # Convert the gamma-corrected image back to uint8 format for values to be integers between 0 to 255
    gamma_corrected = np.uint8(gamma_corrected)

    return gamma_corrected

def main(args: argparse.Namespace) -> None:
    image = cv.imread(str(args.image))
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
