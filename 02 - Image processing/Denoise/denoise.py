import cv2 as cv
import numpy as np

#experimented using all the three types mentioned in md file.
noisy_image = cv.imread('noisy_grayscale.jpg', cv.IMREAD_GRAYSCALE)
image_median = cv.imread('denoised_5x5_median.jpg', cv.IMREAD_GRAYSCALE)

# Apply Median Blur to original noisy image
#kernel size 3x3 looks better than 5x5 but variance with 5x5 filter file is lower for 5x5 kernel size
#7 shows not so good results
median_blur = cv.medianBlur(noisy_image, 5)
cv.imwrite('grayscale_denoised.jpg', median_blur)

# Apply Gaussian Blur to original noisy image
#kernel size 3x3 looks better than 5x5 but variance with 5x5 filter file is lower for 5x5 kernel size
#less blur than median and bilateral
gaussian_blur = cv.GaussianBlur(median_blur, (5, 5), 0)
cv.imwrite('Extra/grayscale_denoised_gaussian.jpg', gaussian_blur)

# Apply Bilateral Filtering to original noisy image
bilateral_filter = cv.bilateralFilter(noisy_image, 9, 75, 75)
cv.imwrite('Extra/grayscale_denoised_bilateral.jpg', bilateral_filter)


#trying to find which method is closest to 5x5 filter jpg file
rmse_g = np.sqrt(np.mean((image_median - gaussian_blur)**2))
rmse_b = np.sqrt(np.mean((image_median - bilateral_filter)**2))
rmse_m = np.sqrt(np.mean((image_median - median_blur)**2))
print(rmse_m, rmse_b, rmse_g)
#1.5354509201489557 5.486729315400374 5.365272181168713

#best one: as per naked eye - guassian blur (3x3)
#best one: as per rmse: median(5x5) -> used this file :grayscale_denoised_median.jpg as final
#cascading inputs doesn't show significant improvement


#experimenting with adaptive filtering
# def adaptive_median_filter(img, max_kernel_size, threshold, median_file):
#     result = np.copy(img)
#     height, width = img.shape
#
#     for y in range(height):
#         for x in range(width):
#             kernel_size = 3  # Start with a 3x3 kernel
#             while kernel_size <= max_kernel_size:
#                 half_kernel = kernel_size // 2
#                 neighborhood = img[max(0, y - half_kernel):min(height, y + half_kernel + 1),
#                                   max(0, x - half_kernel):min(width, x + half_kernel + 1)]
#
#                 median = np.median(neighborhood)
#                 current_pixel = median_file[y,x]
#
#                 if median - current_pixel < threshold:
#                     result[y, x] = median
#                     break  # Stop increasing the kernel size
#                 else:
#                     kernel_size += 2  # Increase the kernel size
#
#     return result

# if __name__ == "__main__":
#     input_image = cv.imread('noisy_grayscale.jpg', cv2.IMREAD_GRAYSCALE)
#     max_kernel_size = 9  # Maximum kernel size to consider
#     threshold = 2  # Adjust this threshold based on your image characteristics
#     image_median = cv.imread('denoised_5x5_median.jpg', cv2.IMREAD_GRAYSCALE)
#
#     # # Apply adaptive median filtering
#     # output_image = adaptive_median_filter(input_image, max_kernel_size, threshold, image_median)
#     #
#
#     # Save the denoised image
#     cv.imwrite('denoised_image.png', output_image)
#
#     # Display the original and denoised images for comparison (optional)
#     cv2.imshow('Original Image', input_image)
#     cv2.imshow('Denoised Image', output_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#     rmse_m = np.sqrt(np.mean((image_median - output_image) ** 2))
#     print(rmse_m)
