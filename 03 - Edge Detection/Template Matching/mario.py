import cv2 as cv
import numpy as np
from helpers import non_maximal_suppression
import matplotlib.pyplot as plt

def main():
    scene = cv.imread("mario.jpg")
    # Read the PNG including the alpha channel in the fourth plane, then split into BGR
    # and alpha parts
    template = cv.imread("coin.png", cv.IMREAD_UNCHANGED)
    print(template.shape)
    template, mask = template[:, :, :3], template[:, :, 3:4]

    # Perform template matching using cv.TM_SQDIFF mode and mask
    result = cv.matchTemplate(scene, template, cv.TM_SQDIFF, mask=mask)

    # max = cv.minMaxLoc(result)

    # Normalize the result to have values between 0 and 255
    #-result inverts the graph
    normalized_result = cv.normalize(-result, None, 0, 255, cv.NORM_MINMAX)
    # max = cv.minMaxLoc(normalized_result)
    # print(max)
    # normalized_result = 255 - normalized_result
    # cv.imwrite("norm.jpg", normalized_result)

    # Plot histogram to choose a threshold
    plt.hist(normalized_result.ravel(), bins=100)
    plt.show()
    print(normalized_result)

    # Set values below the threshold to zero
    mask1 = (normalized_result < 168.8)
    #threshold set to 168 looking at histogram generated
    normalized_result[mask1] = 0
    cv.imshow("Matches", normalized_result)
    cv.waitKey(0)

    # Apply non-maximal suppression to get local maxima
    coordinates = non_maximal_suppression(normalized_result)
    cv.imshow("Matches", coordinates)
    cv.waitKey(0)

    # Get the (x, y) coordinates of nonzero pixels using np.where
    nonzero_coords = np.where(coordinates != 0)

    # Zip the coordinates into (x, y) pairs and get those point counts
    xy_coordinates = list(zip(nonzero_coords[1], nonzero_coords[0]))
    coin_count = len(xy_coordinates)

    # Define text properties (font, color, size, etc.)
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    font_color = (255, 0, 0)  # blue color in BGR format
    font_thickness = 2
    text_x = 10
    text_y = 100
    text = f'Coins Found: {coin_count}'

    for x,y in xy_coordinates:
        cv.rectangle(scene, (x, y), (x + template.shape[1], y + template.shape[0]), (0, 0, 255), 2)
        cv.imshow("scene", scene)
        cv.waitKey(0)
        cv.destroyAllWindows()

    # Use cv.putText() to overlay the text on the image
    cv.putText(scene, text, (text_x, text_y), font, font_scale, font_color, font_thickness)
    # Write image
    cv.imwrite("mario-matches.jpg", scene)

if __name__ == "__main__":
    main()
