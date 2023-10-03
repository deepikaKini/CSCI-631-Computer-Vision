import numpy as np
import cv2 as cv
from pathlib import Path
from helpers import float_to_uint8, uint8_to_float
import argparse


def frequency_coordinates_yx(img_shape: tuple) -> (np.ndarray, np.ndarray):
    """Get the frequency coordinates for an image of a given size. In other words, the outputs are arrays f_y and f_x
    such that f_y[i, j] is the vertical frequency and f_x[i, j] is the horizontal frequency of the sinusoid denoted by
    im_f[i, j] where im_f is the fourier transform of an image of size img_shape.

    For more information, see np.fft.fftfreq.
    """
    return np.meshgrid(np.fft.fftfreq(img_shape[0]), np.fft.fftfreq(img_shape[1]))


def low_pass(img: np.ndarray, cutoff: float) -> np.ndarray:
    """Applies a low pass filter to the image by masking out frequencies higher than cutoff in the fourier domain.

    :param img: the image to filter
    :param cutoff: the cutoff frequency (as a fraction of the width of the image)
    :return: the filtered image
    """
    # Compute the Fourier Transform of the image
    img_f = np.fft.fft2(img, axes=(0, 1))

    # Get the frequency coordinates
    f_y, f_x = frequency_coordinates_yx(img.shape)

    # Calculate the distance from the center of the Fourier domain
    distance = np.sqrt(f_y ** 2 + f_x ** 2)
    print(np.max(img_f), np.min(img_f))
    # Create a mask where values less than cutoff are set to 1 and others to 0
    mask = np.where(distance <= cutoff, 1, 0)
    # Create a mask with the same number of channels as the Fourier transformed image
    mask = np.stack([mask] * 3, axis=-1)
    print(mask)
    print(mask.shape)
    # Apply the mask in the Fourier domain
    img_filtered_f = img_f * mask

    # Take the inverse Fourier Transform and return the real part
    img_filtered = np.fft.ifft2(img_filtered_f, axes=(0, 1)).real
    print(np.max(img_filtered), np.min(img_filtered))
    # img_filtered = (img_filtered * 255).astype(np.uint8)
    print(np.max(img_filtered), np.min(img_filtered))
    return img_filtered



def high_pass(img: np.ndarray, cutoff: float) -> np.ndarray:
    """Applies a high pass filter to the image by masking out frequencies lower than cutoff in the fourier domain.

    :param img: the image to filter
    :param cutoff: the cutoff frequency (as a fraction of the width of the image)
    :return: the filtered image
    """
    # Compute the Fourier Transform of the image
    img_f = np.fft.fft2(img, axes=(0, 1))

    # Get the frequency coordinates
    f_y, f_x = frequency_coordinates_yx(img.shape)

    # Calculate the distance from the center of the Fourier domain
    distance = np.sqrt(f_y ** 2 + f_x ** 2)
    print(np.max(distance), np.min(distance))
    # Create a mask where values greater than cutoff are set to 1 and others to 0
    mask = np.where(distance > cutoff, 1, 0)
    print(mask)
    # cv.imshow("mask", mask)
    # cv.waitKey(0)

    # Create a mask with the same number of channels as the Fourier transformed image
    # mask = np.stack([mask] * 3, axis=-1) --> instead using newaxis to broadcast(taught in class)
    # cv.imshow("mask", mask)
    # cv.waitKey(0)
    print(mask.shape)
    # Apply the mask in the Fourier domain
    img_filtered_f = img_f * mask[:,:, np.newaxis]

    # Take the inverse Fourier Transform and return the real part
    img_filtered = np.fft.ifft2(img_filtered_f, axes=(0, 1)).real
    print(np.max(img_filtered_f), np.min(img_filtered_f))
    # img_filtered = (img_filtered * 255).astype(np.uint8)
    print(np.max(img_filtered), np.min(img_filtered))
    return img_filtered


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image1", type=Path)
    parser.add_argument("image2", type=Path)
    parser.add_argument("--low-cutoff", type=float, required=True)
    parser.add_argument("--high-cutoff", type=float, required=True)
    args = parser.parse_args()

    if not args.image1.exists():
        print(f"Image {args.image1} not found")
        exit(1)

    if not args.image2.exists():
        print(f"Image {args.image2} not found")
        exit(1)

    img1 = uint8_to_float(cv.imread(str(args.image1)))
    img1_low_pass = low_pass(img1, args.low_cutoff)
    cv.imwrite(
        f"output_images/{args.image1.stem}_low_pass.jpg", float_to_uint8(img1_low_pass)
    )

    img2 = uint8_to_float(cv.imread(str(args.image2)))
    img2_high_pass = high_pass(img2, args.high_cutoff)
    cv.imwrite(
        f"output_images/{args.image2.stem}_high_pass.jpg",
        float_to_uint8(img2_high_pass),
    )

    img_hybrid = img1_low_pass + img2_high_pass
    cv.imwrite(
        f"output_images/{args.image1.stem}_{args.image2.stem}_hybrid.jpg",
        float_to_uint8(img_hybrid),
    )
