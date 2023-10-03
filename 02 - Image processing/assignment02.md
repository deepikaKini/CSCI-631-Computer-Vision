Assignment 02 - Image formation / image processing
==================================================

## Problem 1: Getting going with Python, OpenCV, and numpy (18 points, 2 per part)

This assignment is designed to get you familiar with Python, OpenCV, and numpy.

__Anatomy of a command-line Python script:__ If you haven't done much python development yourself, start by opening 
`skeleton.py` to see an example command-line script. This is a common pattern, using the `argparse` module to parse 
command-line arguments, and passing the resulting object to a `main` function, which in turn delegates to other 
user-defined functions.

__Loading and saving images:__ The `skeleton.py` file also demonstrates some basic loading (`cv.imread`) and saving 
(`cv.imwrite`) operations with images.

__Displaying images and getting user input:__ In addition to reading and writing, OpenCV has some built-in tools for 
displaying and interacting with images. To display an image, you can do the following:

```python
# Display image
cv.imshow('window title', image)
# Wait for the user to press the 'Q' key
while True:
  k = cv.waitKey(1)
  if k == ord('q'):
    break
# Close all windows
cv.destroyAllWindows()
```

However, note that this will not work if you run things remotely over SSH. There are a few ways to see images over 
SSH, including X11 forwarding, installing plugins to your IDE, or just `imwrite` to save an image and `scp` to copy 
it back to your machine.

The above example shows how to get keyboard input from the user with the `waitKey` function. This function returns 
the ASCII code of the key that was pressed, or -1 if no key was pressed. You can use the `ord` function to convert a 
character to its integer representation. For example, `ord('q')` returns 113. (The inverse or `ord()` is `chr()`).

To get mouse input, you can use the `setMouseCallback` function to register a callback function that will be called 
when the user clicks on the image. The callback function should have the following signature:

```python
def callback(event, x, y, flags, param):
  # event is one of cv.EVENT_* constants
  # x and y are the coordinates of the mouse click
  # flags is a bitfield of cv.EVENT_FLAG_* constants
  # param is a user-defined parameter passed to setMouseCallback
```

Part 1/9: copy `skeleton.py` to a new file called `coordinates.py`. Modify it to display the image and 
wait for the user to press the 'Q' key before exiting. Have it print the x, y coordinates of wherever a user clicks on the image. Below, write the x, y coordinates of the dragonfly (on a flower) and the seashell (on the table) from the `bouquet.jpg` image.

    coordintes of the dragonfly: 456, 458
    coordintes of the seashell: 733, 1498
    does the x coordinate index rows (distance from the top) or columns (distance from the left)? cols
    does the y coordinate index rows (distance from the top) or columns (distance from the left)?rows

__Slicing and indexing numpy arrays:__ Images are stored as numpy arrays. Let's see how to slice and index one. This 
part of the assignment has a few steps. Open `image_manipulation.py` and implement functions for the following. 
Again using the `bouquet.jpg` file as the input, do the following:

1. Part 2/9: crop the image to a 100x100 square centered on a point the user clicks. Use this to crop an image around the  dragonfly. Save the result as `clicked.jpg`. To do this, you can use the `image[a:b, c:d]` syntax to slice the 
   array between rows `a` and `b` and columns `c` and `d`. Note that specifying indices on the third (BGR) dimension 
   is optional, so `image[a:b, c:d, :]` is equivalent. Be mindful of Python's indexing conventions – submissions with a 99x99 or 101x101 crop will be marked wrong.
   
2. Part 3/9: scale the image down to half size on both rows and columns using numpy slicing. Save the result as `halfsize.jpg`.(Technically, it is a quarter of the original number of pixels). To do this, you can use the `x[::n]` syntax to select every `n`th element of an array.

3. Part 4/9: scale the image down to half size on both rows and columns using OpenCV. Save the result as `halfsize_cv.jpg`. To do this, you can use the `cv.resize` function.
   
4. Part 5/9: flip the image horizontally using numpy slicing. Save the result as `flipped.jpg`. To do this, you can use the `x[::-1]` syntax to reverse the order of an array (but you'll need to figure out horizontal versus vertical slicing), or by using the `np.fliplr` function.

5. Part 6/9: rotate the image 90 degrees counterclockwise using the `np.transpose` or `np.permute` function. Save the result as `rotated.jpg`.

6. Part 7/9: swap the B and R channels of the image. Save the result as `swapped.jpg`. To do this, you can use the `image[:,:,i]`syntax to select the `i`th channel of the image, or `cv.split` and `cv.merge`
7. Part 8/9: convert the image to LAB color space and invert the hue by replacing the `a` channel with `255 - a` and the `b`
   channel with `255 - b`. Then, convert back to BGR color space. Save the result as `inverted_ab.jpg`. To do this, you
   will need the `cv.cvtColor` function as well as `cv.split` and `cv.merge`.

What differences do you notice between `halfsize.jpg` and `halfsize_cv.jpg`? What does this tell you about how `cv.
resize()` works when called with default parameters?

Answer (part 9/9): __cv image looks more blurred compared to non-cv method at some regions of the image.
the resize method: parameters are the src, dsize,(size required), fx(scaling on x) and fy(scaling on y). The first and any one from other 3  are required.
The last optional parameter which is interpolation is the technique used to decrease the size by using computations on neighnour values. Default is INTER_LINEAR which works avergely well for applications. Other values may improve quality but may be slower or vice versa.__

__Tips:__
- Apart from the functions that require user input, this can be run locally on your computer or on one of the CS 
  servers. If you want to run things from PyCharm, you can set the command line arguments in the Run/Debug 
  configuration. I assume something similar is available in other IDEs, but if you are using something else, you 
  will have to figure it out.
- Slicing a numpy array sometimes results in a _view_ of the array rather than a _copy_. OpenCV functions do not work 
  with views, so you may need to use the numpy `copy` method to make a copy of the array, such as 
  `image = image[:,::-1].copy()`
- Images loaded in OpenCV are stored in BGR not RGB format
- Values in the range 0-255 are stored as `uint8` data type. You can use the `x.astype(np.uint8)` function to convert 
  a numpy array `x` to `uint8` data type.
- You can use the `cv.split` and `cv.merge` functions to split and merge color channels
- You can use the `cv.cvtColor` function to convert between color spaces



## Problem 2: Rebalance colors in underwater photography (10 points)

Within the visible spectrum, long wavelengths of light are absorbed more in water than short wavelengths. This gives 
photos taken underwater a strong bluish tint and low overall contrast. This assignment includes two photos taken 
underwater - `underwater01.jpg` and `underwater02.jpg`. Your task is to perform some color adjustments using numpy and OpenCV 
to make them look better. Copy `skeleton.py` to `underwater.py`. Save the results as `underwater01_adjusted.jpg` and 
`underwater02_adjusted.jpg`.  For each image, try the following:

1. Scale the red channel by 120% and the blue channel by 80%.
2. Convert from BGR to LAB
3. Equalize the histogram of the L channel to improve contrast (or partially equalize if it looks better to you)
4. Do whatever other operations you think will improve the image
5. Return the image in BGR format for saving to a new image file

You do not need to follow these steps exactly. This problem will be graded based on the quality of the output images
and the thoughtfulness of your approach. Creative approaches that deviate from the above script are encouraged. For
example, you may get better results by undoing gamma correction before performing color adjustments and then redoing
gamma correction before saving the image (see CVAA section 2.3.2 for more information on gamma correction).

__Text response:__ These images were taken with a standard GoPro used as a point-and-shoot camera underwater. Given what
you learned about digital image sensors and properties of light, how would you improve the image quality of such 
underwater photos? There are limits to what can be done in post-processing, as you've just seen. Think about modifying
the sensor or bringing additional equipment underwater with you.


__Answer:__
1. Can improve the lighting,  maybe use flash.
2. lower ISO settings that help to improve sensitivity and use a red filter to decrease blue light dominating.

__Tips:__

- You can use the `cv.equalizeHist` function to perform histogram equalization or, for more of a challenge but more
  control and flexibility, you can build your own lookup table and use `cv.LUT`

## Problem 3: Image denoising (12 points)

This is essentially exercise 3.12 or 3.14 from the CVAA textbook, but we'll focus on noise removal and get to other
kinds of filters next week.

The image `grayscale_noise.jpg` is a grayscale image with Gaussian noise added to each pixel. (If you're curious, it was
generated using the `generate_noisy_grayscale.py` file that has also been provided to you for reference, though you are
not given access to the original photo directly). Your task is to remove the noise as well as possible. Since the noise
was added artificially, we have access to the ground truth and can measure the quality of your result numerically. The
closer you get to recovering the original image, the higher your score on this problem. As long as your code runs and 
makes a good-faith attempt at noise removal, you will get at least 5 points. The remaining 10 points will be awarded
based on the quality of your result. The file `denoised_5x5_median.jpg` shows results when the image is adjusted using
a 5x5 median filter. If your method does at least as well as this, you will get at least 10 total points. The last 2 
points for this problem will be awarded based on how well your method does relative to your classmates. (2 points will
matter very little for your final grade - this is just to add some friendly competition.)

Write your code in `denoise.py` and save the result as `grayscale_denoised.jpg`.

Some ideas to try:
- Median blur (as in the example)
- Gaussian blur
- Bilateral filtering (section 3.3.2)

## Problem 4: Pyramid blending (12 points)

1. (5 points) Do exercise 3.18 from the CVAA book. First, recreate Figure 3.41(d) from the book using `apple.png` and 
   `orange.png` that have been provided to you. [This website](https://docs.opencv.org/4.x/dc/dff/tutorial_py_pyramids.html)
   shows how to do this exercise using OpenCV. Save the result as `apple_orange_blended.jpg`.
   __(python file: blend_apple_orange.py, mask file: mask_ap.jpg)__
   
2. (7 points) Second, create your own. Find (or take) two images that you think would blend well together. Create a binary mask as shown in Figure 3.41(g). Blend them together using the same method as above. Save the input images as `source1.jpg` and `source2.jpg`, the mask as `mask.jpg`, and the blended result as `blended.jpg`.
__(python file: blend_4b.py)__

## Collaboration and Generative AI disclosure

Did you collaborate with anyone? Did you use any Generative AI tools? Briefly explain what you did here.
__Collaborated with a few people to understand if they were able to use mask for question4.
Used ChatGPT to get answers to theoretical questions and about improvement in code during errors showing up__

## About how long did you spend on this assignment?
__About 3 day productivity hours. Around 16 hours.__

This is optional to disclose, but it helps us improve the course if you can estimate for us how long this assignment 
took for you.

## Submitting

Your submission should consist of a single zip file named like `firstname_lastname_assignment2.zip`. The zip file 
should contain all the contents of this directory, including this file (which you should have written some answers 
into above) and any images or other outputs generated by your code.



=========Personal notes=======

Coordinates: x=456, y=458
Coordinates: x=6, y=8
Coordinates: x=1192, y=10
Coordinates: x=1169, y=1504
Coordinates: x=26, y=1501
