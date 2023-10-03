import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def take_picture():
    cam = cv.VideoCapture(0)
    if not cam.isOpened():
        print("Cannot open camera")
        exit()

    # Read image from camera. Loop a few times to give the camera a moment to "wake up"
    # and adjust to lighting conditions
    image = None
    for _ in range(5):
        res, image = cam.read()
        if not res:
            print("Cannot read camera")
            exit()

    cam.release()
    return image


def plot_channels_histogram(image):
    # Plot histogram of BGR channels' intensities
    b, g, r = cv.split(image)
    hist_b = cv.calcHist([b], [0], None, [256], [0, 256])
    hist_g = cv.calcHist([g], [0], None, [256], [0, 256])
    hist_r = cv.calcHist([r], [0], None, [256], [0, 256])
    plt.plot(hist_b, color='b')
    plt.plot(hist_g, color='g')
    plt.plot(hist_r, color='r')
    plt.xlabel("Intensity")
    plt.ylabel("Count")
    plt.show()


def photo_negative(image):
    # Create negative of image
    return 255 - image


def blur(image):
    return cv.GaussianBlur(image, (15, 15), 0)


def posterize(image):
    # Blur then posterize image
    blurred = blur(image)
    return blurred // 64 * 64


def main():
    image = take_picture()

    # Tile image with some modified versions of itself and show it
    negative = photo_negative(image)
    blurred = blur(image)
    posterized = posterize(image)

    final_image = np.concatenate((
        np.concatenate((image, negative), axis=1),
        np.concatenate((posterized, blurred), axis=1)),
        axis=0)

    cv.putText(final_image,
               "Hello World!",
               (final_image.shape[1]//2, final_image.shape[0]//2),
               cv.FONT_HERSHEY_SIMPLEX,
               5,
               (0, 255, 0),
               5,
               cv.LINE_AA)

    cv.imwrite("hello.jpg", final_image)

    cv.imshow("hello.jpg", final_image)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
