import cv2 as cv
import numpy as np
import warnings


def uint8_to_float(image: np.ndarray) -> np.ndarray:
    """Converts a uint8 image to a float image in the range [0.0, 1.0].
    """
    if image.dtype.kind == 'f':
        return image
    elif image.dtype == np.uint8:
        return image / 255.0
    else:
        raise TypeError(f"Unsupported image type: {image.dtype}")


def float_to_uint8(image: np.ndarray) -> np.ndarray:
    """Converts a float image in the range [0.0, 1.0] to a uint8 image.
    """
    if image.dtype == np.uint8:
        return image
    if image.dtype.kind == 'f':
        return np.clip(np.round(image * 255), 0, 255).astype(np.uint8)
    else:
        raise TypeError(f"Unsupported image type: {image.dtype}")


def conv2D(f: np.ndarray,
           h: np.ndarray,
           borderType: int = cv.BORDER_REPLICATE) -> np.ndarray:
    """2D convolution (whereas cv.filter2D is correlation). Convolution is equivalent to correlation with a flipped
    filter.
    """
    f, h = uint8_to_float(f), uint8_to_float(h)
    return cv.filter2D(f, -1, cv.flip(h, -1), borderType=borderType)


def filter_filter(h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
    """Return a new filter such that convolving with h1 then convolving with h2 is equivalent to just convolving with
    the combined filter.

    In other words, return h3 such that
        h3 = filter_filter(h1, h2).
        conv2D(conv2D(f, h1), h2) == conv2D(f, h3)

    Under the hood, this function makes use of the fact that convolution is associative and constructs h3 by
    convolving h1 with h2.
    """
    if h1.ndim != 2 or h2.ndim != 2:
        raise ValueError("Filters must be 2D (but they may be column or row vectors).")
    # Need to pad h1 with zeros on each side to "make room" for the filter h2
    m, n = h2.shape[:2]
    h1_padded = cv.copyMakeBorder(h1,
                                  top=m//2,
                                  bottom=m//2,
                                  left=n//2,
                                  right=n//2,
                                  borderType=cv.BORDER_CONSTANT,
                                  value=[0.0])
    return conv2D(h1_padded, h2)
