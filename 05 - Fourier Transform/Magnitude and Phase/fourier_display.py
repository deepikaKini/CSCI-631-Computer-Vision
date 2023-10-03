import cv2 as cv
import numpy as np
from pathlib import Path
import argparse


def fourier_magnitude_as_image(img_f: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Displays the log magnitude of the fourier transform of the image with the DC component in the center.

    :param img_f: fourier transform of an image
    :return: a BGR or grayscale image representing the (log) magnitude of the fourier transform.
    """
    #shift 0 components on both axes
    img_f_shifted = np.fft.fftshift(img_f)

    # Compute the magnitude and normalize it
    #magnitude is absolute values of real and imaginary of F(u): squareroot of summation of squares
    magnitude = np.log(eps + np.abs(img_f_shifted))
    magnitude = (magnitude - np.min(magnitude)) / (np.max(magnitude) - np.min(magnitude)) * 255

    # Convert the magnitude to uint8
    magnitude = magnitude.astype(np.uint8)
    print(np.max(magnitude), np.min(magnitude))
    # Create a BGR image if the input is grayscale
    # if len(img_f.shape) == 2:
    magnitude_img = cv.cvtColor(magnitude, cv.COLOR_GRAY2BGR)
    # else:
    #     magnitude_img = magnitude

    return magnitude_img


def fourier_phase_as_image(img_f: np.ndarray) -> np.ndarray:
    """Displays the phase of the fourier transform of the image with the DC component in the center, using HSV color
    space (saturation and value both set to max).

    :param img_f: fourier transform of an image
    :return: a BGR image representing the phase of the fourier transform with hue.
    """
    # shift 0 components on both axes
    img_f_shifted = np.fft.fftshift(img_f)
    #compute phase by using np.angle (uses arctan2 with division of imaginary and real component)
    phase = np.angle(img_f_shifted)
    #Convert values between 0 to 180 (since it is in radians)
    phase = (phase - np.min(phase)) / (np.max(phase) - np.min(phase)) * 180
    # phase = np.degrees(phase)
    print(np.max(phase),np.min(phase))
    # Set saturation and value both to 255 and hue to the phase value
    hsv_image = np.zeros((img_f.shape[0], img_f.shape[1], 3), dtype=np.uint8)
    hsv_image[..., 0] = phase
    hsv_image[..., 1] = 255
    hsv_image[..., 2] = 255

    # Convert the HSV image to BGR
    phase_img = cv.cvtColor(hsv_image, cv.COLOR_HSV2BGR)

    return phase_img

def make_mag_phase_images(image_file: Path):
    file_mag = Path("output_images") / f"{image_file.stem}_mag.png"
    file_phase = Path("output_images") / f"{image_file.stem}_phase.png"

    img = cv.imread(str(image_file), cv.IMREAD_GRAYSCALE)

    # Do the fourier transform
    img_f = np.fft.fft2(img)

    # Get mag and phase and write those images
    img_mag = fourier_magnitude_as_image(img_f)
    cv.imwrite(str(file_mag), img_mag)

    img_phase = fourier_phase_as_image(img_f)
    cv.imwrite(str(file_phase), img_phase)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("images",
                        help="One or more images (space-separated if more than one) to process",
                        type=Path,
                        nargs="+")
    args = parser.parse_args()

    for image in args.images:
        if not image.exists():
            print(f"Image {image} not found")
            continue

        make_mag_phase_images(image)
