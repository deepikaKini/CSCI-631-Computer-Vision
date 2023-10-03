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


def non_maximal_suppression(values, out=None, window=5):
    """Performs non-maximal suppression on the given values matrix. The window size is the size of the neighbourhood
    to consider around each pixel.

    The values matrix is not modified, but a new matrix is returned, or the 'out' matrix is used if provided.
    """
    out = out or values.copy()
    if values.ndim == 3:
        h, w, c = values.shape
        if c != 1:
            raise ValueError("Values matrix must be 2D or 3D with a single channel.")
    else:
        h, w = values.shape[:2]
    for y in range(h):
        for x in range(w):
            # Get the neighbourhood around the current pixel
            y1 = max(0, y - window // 2)
            y2 = min(h, y + window // 2 + 1)
            x1 = max(0, x - window // 2)
            x2 = min(w, x + window // 2 + 1)
            neighbourhood = values[y1:y2, x1:x2]
            # Set the pixel to min value if it is not the maximum in the neighbourhood
            out[y, x] = 0 if values[y, x] < neighbourhood.max() else values[y, x]
    return out


def conv2D(f: np.ndarray,
           h: np.ndarray,
           borderType: int = cv.BORDER_REPLICATE) -> np.ndarray:
    """2D convolution (whereas cv.filter2D is correlation). Convolution is equivalent to correlation with a flipped
    filter.
    """
    f, h = uint8_to_float(f), uint8_to_float(h)
    if is_separable(h):
        u, v = split_separable_filter(h)
        f_u = cv.filter2D(f, -1, u, borderType=borderType)
        return cv.filter2D(f_u, -1, v, borderType=borderType)
    else:
        return cv.filter2D(f, -1, cv.flip(h, -1), borderType=borderType)


def split_separable_filter(h:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Given a separable filter, return the two 1D filters that make it up. If the given filter is not separable,
    this function returns the best approximation of it and displays a warning.

    Returns: (vertical_part, horizontal_part) where each is a 1D filter. The vertical part is a column vector and the
    horizontal part is a row vector.
    """
    if h.ndim != 2:
        raise ValueError("Filter must be 2D.")
    if not is_separable(h):
        warnings.warn("Filter is not separable. Using best approximation.")
    u, s, vh = np.linalg.svd(h)
    vertical_part = u[:, :1] * np.sqrt(s[:1])
    horizontal_part = vh[:1, :] * np.sqrt(s[:1])
    return vertical_part, horizontal_part


def is_separable(h:np.ndarray, tolerance: float = 1e-6) -> bool:
    """Return whether a given 2D filter is separable within a given tolerance.
    """
    if h.ndim != 2:
        raise ValueError("Filter must be 2D.")
    u, s, vh = np.linalg.svd(h)
    sum_of_singular_values = np.sum(s[1:])
    return sum_of_singular_values < tolerance


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
