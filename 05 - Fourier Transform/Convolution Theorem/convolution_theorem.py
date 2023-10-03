import numpy as np
import cv2 as cv
from pathlib import Path

import helpers
from helpers import conv2D
import argparse


def load_filter(filter_file: str) -> np.ndarray:
    """Load a filter from a plain text file into a 2D numpy array."""
    with open(filter_file, "r") as file:
        lines = file.readlines()
        filter_data = [list(map(float, line.strip().split())) for line in lines]
        return np.array(filter_data)

def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image values to the range [0, 255] and convert to uint8."""
    image_normalized = ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).astype(np.uint8)
    image_normalized  = helpers.float_to_uint8(image_normalized)
    return image_normalized

def convolution_spatial(f: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Perform convolution in the spatial domain using helper functions."""
    return conv2D(f, h)

# def convolution_frequency(f: np.ndarray, h: np.ndarray) -> np.ndarray:
#     """Perform convolution in the frequency domain."""
#     # Compute the Fourier Transform of the image
#     ft_f = np.fft.fft2(f, axes=(0, 1))
#
#     # Compute the Fourier Transform of the filter and pad it to the image size
#     ft_h = np.fft.fft2(h, s=f.shape[:2], axes=(0, 1))
#     m, n = ft_f.shape[:2]
#     ft_h = cv.copyMakeBorder(ft_h,
#                                   top=m // 2,
#                                   bottom=m // 2,
#                                   left=n // 2,
#                                   right=n // 2,
#                                   borderType=cv.BORDER_CONSTANT,
#                                   value=[0.0])
#
#     # Multiply the Fourier Transforms
#     result = ft_f * ft_h
#
#     # Take the real part of the inverse Fourier Transform
#     result = np.fft.ifft2(result, axes=(0, 1)).real
#
#     return result
def convolution_frequency_single_channel(f_f: np.ndarray, h_f: np.ndarray) -> np.ndarray:
    """Perform convolution in the frequency domain for a single channel."""
    # Compute the Fourier Transform of the image



    # filter_f = np.fft.fft2(h, axes=(0, 1))


    # Multiply the Fourier Transforms
    result_f = f_f * h_f
    # result_f = np.fft.ifftshift(result_f, (0,1))

    # Take the real part of the inverse Fourier Transform
    result = np.fft.ifft2(result_f, axes=(0, 1)).real

    return result

def convolution_frequency(f: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Perform convolution in the frequency domain."""
    # Initialize an empty result image
    result = np.zeros_like(f, dtype=np.float64)
    #


    f_f = np.fft.fft2(f, axes=(0, 1))

    # # shift 0 components on both axes
    # img_f_shifted = np.fft.fftshift(f_f)
    #
    # # Compute the magnitude and normalize it
    # # magnitude is absolute values of real and imaginary of F(u): squareroot of summation of squares
    # magnitude = np.log( 1e-6 + np.abs(img_f_shifted))
    # magnitude = (magnitude - np.min(magnitude)) / (np.max(magnitude) - np.min(magnitude)) * 255
    #
    # # Convert the magnitude to uint8
    # magnitude = magnitude.astype(np.uint8)
    # print(np.max(magnitude), np.min(magnitude))
    # # Create a BGR image if the input is grayscale
    # # if len(img_f.shape) == 2:
    # # magnitude_img = cv.cvtColor(magnitude, cv.COLOR_GRAY2BGR)
    # cv.imwrite("mag.jpg", magnitude)



    # Compute the Fourier Transform of the filter and pad it to the image size
    h_f = np.fft.fft2(h, s=f.shape[:2], axes=(0, 1))

    # Apply convolution to each channel separately
    for channel in range(3):
        result[:, :, channel] = convolution_frequency_single_channel(f_f[:, :, channel],h_f)

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--filter", type=Path, required=True)
    parser.add_argument("--mode", choices=["spatial", "frequency"], default="spatial")
    args = parser.parse_args()

    if not args.image.exists():
        print(f"Image {args.image} not found")
        exit(1)

    if not args.filter.exists():
        print(f"Filter {args.filter} not found")
        exit(1)

    # Load the image
    image = cv.imread(str(args.image))

    # Load the filter
    filter_matrix = load_filter(str(args.filter))
    # m, n = image.shape[:2]
    # filter_matrix = cv.copyMakeBorder(filter_matrix,
    #                               top=m // 2,
    #                               bottom=m // 2,
    #                               left=n // 2,
    #                               right=n // 2,
    #                               borderType=cv.BORDER_CONSTANT,
    #                               value=[0.0])

    # Perform convolution based on the mode
    if args.mode == "spatial":
        result = convolution_spatial(image, filter_matrix)
    else:
        result = convolution_frequency(image, filter_matrix)

    print("result has dtype", result.dtype)

    # Normalize and save the result image
    result_normalized = normalize_image(result)
    # result_normalized = result_normalized[:,:, 0]
    # result_normalized = cv.cvtColor(result_normalized, cv.COLOR_BGR2GRAY)
    output_path = f"output_images/{args.image.stem}_filtered.jpg"
    cv.imwrite(str(output_path), result_normalized)
