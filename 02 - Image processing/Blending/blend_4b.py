import cv2
import numpy as np
import cv2


# generate Laplacian Pyramid for A
def generate_laplacian_pyramid(levels, gpA):
    lpA = [gpA[levels]]
    for i in range(levels, 0, -1):
        GE = cv2.pyrUp(gpA[i], dstsize= (gpA[i-1].shape[1], gpA[i-1].shape[0]))
        # print(GE.shape)
        # print(gpA[i - 1].shape)
        # GE = cv2.resize(GE, gpA[i - 1].shape[1::-1])
        # print(GE.shape)
        L = cv2.subtract(gpA[i - 1], GE)
        lpA.append(L)
    return lpA


def generate_gaussian_pyramid(image, levels):
    # generate Gaussian pyramid for A
    G = image.copy()
    gpA = [G]
    for i in range(levels):
        G = cv2.pyrDown(G)
        gpA.append(G)
    return gpA

def pyramid_blend(image1, image2, mask):
    levels = 5  # pyramid levels count

    # Generate Gaussian pyramids for image1, image2, mask and inverse
    g_p_a = generate_gaussian_pyramid(image1, levels)
    g_p_b = generate_gaussian_pyramid(image2, levels)
    g_p_m = generate_gaussian_pyramid(mask, levels)
    g_p_m_invert = generate_gaussian_pyramid(1 - mask, levels)

    # Generate Laplacian pyramids for image1, image2
    image1_pyramid = generate_laplacian_pyramid(levels, g_p_a)
    image2_pyramid = generate_laplacian_pyramid(levels, g_p_b)

    blended_pyramid = []
    for i in range(levels + 1):
        g_p_m[i] = cv2.resize(g_p_m[i], (image1_pyramid[i].shape[1], image1_pyramid[i].shape[0]))
        g_p_m_invert[i] = cv2.resize(g_p_m_invert[i], (image2_pyramid[i].shape[1], image2_pyramid[i].shape[0]))

        # Perform row-wise multiplication
        calc1 = g_p_m[i] * image1_pyramid[i]
        calc2 = g_p_m_invert[i] * image2_pyramid[i]

        # Calculate the blended level
        blended_level = calc1 + calc2
        blended_pyramid.append(blended_level)

    # Reconstruct the final blended image from the blended pyramid
    blended_image = blended_pyramid[0]
    for i in range(1, levels):

        row = cv2.pyrUp(blended_image, dstsize=(blended_pyramid[i].shape[1],blended_pyramid[i].shape[0]))
        # Perform pyramid upscaling with proper dimensions
        blended_image  = cv2.add(row, blended_pyramid[i] )

    return blended_image


def generate_mask(image):
    height, width, _ = image.shape
    mask = np.zeros((height, width), dtype=np.uint8)

    # Define the center and radius of the circle
    mask[ height//4: height//2 + 100,:] = 1


    cv2.imwrite('mask.png', mask)

# Load two color images and a binary mask
image1 = cv2.imread('source1.JPG')
image2 = cv2.imread('source2.jpg')
image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

mask = generate_mask(image1)
mask = cv2.imread('mask.png')
mask = cv2.resize(mask, (image1.shape[1], image1.shape[0]))
# Ensure images have the same dimensions
common_size = (image1.shape[1], image1.shape[0])
image2 = cv2.resize(image2, common_size)


# Perform pyramid blending
blended_image = pyramid_blend(image1, image2, mask)
#
# Display or save the blended image
cv2.imwrite('blended.jpg', blended_image)

