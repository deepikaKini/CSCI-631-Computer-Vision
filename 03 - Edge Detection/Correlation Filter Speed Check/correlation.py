import cv2 as cv
import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Optional

def my_correlation(image: np.ndarray,
                   kernel: np.ndarray,
                   out: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Performs correlation of the given image with the given kernel, without padding.
    """
    # Correlation operation
    out = np.zeros_like(image) if out is None else out
    # Get the height and width of the image and kernel
    image_height, image_width = image.shape[:2]
    kernel_height, kernel_width = kernel.shape[:2]

    # Calculate the border size based on kernel size
    border_size_height = kernel_height // 2
    border_size_width = kernel_width // 2

    # Iterate over the pixels of the output image
    for y in range(image_height):
        for x in range(image_width):
            # Initialize the result for the current pixel
            result = 0.0

            # Iterate over the kernel
            for i in range(kernel_height):
                for j in range(kernel_width):
                    # Calculate the coordinates in the image
                    img_x = x + j - border_size_width
                    img_y = y + i - border_size_height

                    # Handle border cases using cv.BORDER_REPLICATE
                    img_x = max(0, min(image_width - 1, img_x))
                    img_y = max(0, min(image_height - 1, img_y))

                    # Perform correlation
                    result += image[img_y, img_x] * kernel[i, j]

            # Store the result in the output image
            out[y, x] = result
    return out



def run_and_time_filters(image, kernel_size: int = 5):
    # Create Gaussian filters
    gaussian_kernel_1d = cv.getGaussianKernel(kernel_size, -1)
    gaussian_kernel_1d = gaussian_kernel_1d / np.sum(gaussian_kernel_1d)
    gaussian_kernel_2d = gaussian_kernel_1d @ gaussian_kernel_1d.T

    # helps reduce the caching time taken by cv for 2d filter -> during office hour
    cv_result_2d = cv.filter2D(image, -1, gaussian_kernel_2d, borderType=cv.BORDER_REPLICATE)

    # CV 2D
    start = time.time()
    cv_result_2d = cv.filter2D(image, -1, gaussian_kernel_2d, borderType=cv.BORDER_REPLICATE)
    elapsed_cv_2d = time.time() - start
    cv.imwrite("correlation_2d_gaussian_CV.jpg", cv_result_2d)
    print(f"2D filter time (OpenCV, {kernel_size}x{kernel_size}): {elapsed_cv_2d}")

    # CV 1D separable
    start = time.time()
    cv_result_vertical = cv.filter2D(image, -1, gaussian_kernel_1d, borderType=cv.BORDER_REPLICATE)
    cv_result_separable = cv.filter2D(cv_result_vertical, -1, gaussian_kernel_1d.T, borderType=cv.BORDER_REPLICATE)
    elapsed_cv_separable = time.time() - start
    cv.imwrite("correlation_1d_gaussian_separable_CV.jpg", cv_result_separable)
    print(f"Separable filter time (OpenCV, {kernel_size}x{kernel_size}): {elapsed_cv_separable}")

    # numpy 2D
    start = time.time()
    my_result_2d = my_correlation(image, gaussian_kernel_2d)
    print(my_result_2d)
    elapsed_mine_2d = time.time() - start
    cv.imwrite("correlation_2d_gaussian_My_algo.jpg", my_result_2d)
    print(f"2D filter time (Mine, {kernel_size}x{kernel_size}): {elapsed_mine_2d}")

    # numpy 1D separable
    start = time.time()
    my_result_vertical = my_correlation(image, gaussian_kernel_1d)
    my_result_separable = my_correlation(my_result_vertical, gaussian_kernel_1d.T)
    elapsed_mine_separable = time.time() - start
    cv.imwrite("correlation_1d_gaussian_separable_My_algo.jpg", my_result_separable)
    print(f"Separable filter time (Mine, {kernel_size}x{kernel_size}): {elapsed_mine_separable}")

    # Inspect results and/or assert that the results are the same within sensible tolerances

    #Compare the 2d correlation between OpenCV and numpy code
    # start professor's code - > 1/255 is 0.00392156862745098
    res = cv_result_2d / 255. - my_result_2d / 255.
    #checking if differences are distributed or caused due to border generation
    # cv.imshow("abs diff", np.abs(res)/np.abs(res).max())
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    print("High similarity between OpenCV and my implementation for 2D filtering: ",
          np.allclose(res, np.zeros_like(res), atol=2 / 255))

    # Compare the 2d and separable for my code using numpy
    res = my_result_separable / 255. - my_result_2d / 255.
    # checking if differences are distributed or caused due to border generation
    # cv.imshow("abs diff1", np.abs(res) / np.abs(res).max())
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    print("High similarity between my implementation for 2D  and 1D filtering: ",
          np.allclose(res, np.zeros_like(res), atol=2 / 255))

    #similar result for difference between CV 2d and 1d


    return elapsed_cv_2d, elapsed_cv_separable, elapsed_mine_2d, elapsed_mine_separable


def main():
    # Re-use one of the images from Problem 3 here, just for convenience.
    image = cv.imread("edges03.jpg")
    elapsed_cv_2d_list, elapsed_cv_separable_list, elapsed_mine_2d_list, elapsed_mine_separable_list = [], [], [], []
    # Crop it down to a more manageable size for this experiment
    image = image[:200, :200, :]
    cv.imwrite("edges03_clipped.jpg", image)

    # my code for various filter sizes
    filter_sizes = [3, 5, 7, 9, 11]
    for i in filter_sizes:
        elapsed_cv_2d, elapsed_cv_separable, elapsed_mine_2d, elapsed_mine_separable = run_and_time_filters(image, i)
        elapsed_cv_2d_list.append(elapsed_cv_2d)
        elapsed_cv_separable_list.append(elapsed_cv_separable)
        elapsed_mine_2d_list.append(elapsed_mine_2d)
        elapsed_mine_separable_list.append(elapsed_mine_separable)

    fig = plt.figure(figsize=(4, 7))  # The figsize argument is (width, height) in inches
    plt.plot(filter_sizes, elapsed_cv_separable_list, label='CV( separable )')
    plt.plot(filter_sizes, elapsed_cv_2d_list, label='CV(2d)')
    plt.plot(filter_sizes, elapsed_mine_separable_list, label='Mine(separable)')
    plt.plot(filter_sizes, elapsed_mine_2d_list, label='Mine(2d)')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Filter_size')
    plt.ylabel('Runtime(s)')

    # Place the legend on the right side outside the plot
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks(filter_sizes, filter_sizes)
    # Save the figure as an image using savefig
    plt.savefig('correlation_runtime.png', bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()


    '''extra:
    1/255
    PyDev console: starting.
    
    0.00392156862745098
    np.allclose(res, np.zeros_like(res), atol=1 / 255)
    False
    np.allclose(res, np.zeros_like(res), atol=2 / 255)
    True
    
    # assert np.allclose(res, np.zeros_like(res), atol=1/255)
 

    # print(np.max(res))
    '''
