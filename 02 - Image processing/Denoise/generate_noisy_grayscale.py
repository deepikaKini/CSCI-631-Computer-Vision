import cv2 as cv
import numpy as np
import argparse
from pathlib import Path
import os


def main(args: argparse.Namespace) -> None:
    image = cv.imread(str(args.image))
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imwrite("grayscale_ground_truth.jpg", image)
    image = image.astype(np.float32)
    image = image + np.random.randn(*image.shape) * args.noise
    image = np.clip(image, 0, 255).astype(np.uint8)
    cv.imwrite("noisy_grayscale.jpg", image)


# Code inside the if statement below will only be executed if the script is called
# directly from the command line
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--noise", type=float, default=15.0)
    args = parser.parse_args()

    if not args.image.exists():
        print(f"File {args.image} not found")
        exit(1)

    main(args)
