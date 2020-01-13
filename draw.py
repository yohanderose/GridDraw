"""
A useful script to process reference photos for sketches.
Use the arrow keys to shift the ref grid over the image
and the +/- keys (without control or shift) to change the
degree the image is posterised (K changes are hard to notice).

Author: Yohan de Rose

TODO:
- Clicking stops functionality
- Moving grid far enough will show empty areas
- More cmd line control
- Saving project dir with two grid images with and without poster
"""

import cv2
import numpy as np
import sys
import tty
import termios

xLines = []
yLines = []
SHIFT_SIZE = 10
GRID_SIZE = 50
K = 8


def posterise(img):
    print("Posterising image using K of", K, "...")

    Z = img.reshape((-1, 3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    ret, label, center = cv2.kmeans(
        Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    print("Done.")

    return res2


def shiftGrid(img, axis, offset):
    if axis == 0:
        for line in xLines:
            line[0][0] += offset
            line[1][0] += offset
    else:
        for line in yLines:
            line[0][1] += offset
            line[1][1] += offset

    return drawGrid(img)


def drawGrid(img):

    for line in xLines:
        cv2.line(img, tuple(line[0]), tuple(line[1]), (255, 0, 0), 1, 1)

    for line in yLines:
        cv2.line(img, tuple(line[0]), tuple(line[1]), (255, 0, 0), 1, 1)

    return img


def draw(img):
    regions = posterise(img)
    return drawGrid(regions.copy()), regions


def main():
    global K

    img = cv2.imread(sys.argv[1])

    height, width, channels = img.shape

    for x in range(0, width - 1, GRID_SIZE):
        line = [[x, 0], [x, height]]
        xLines.append(line)

    for y in range(0, height - 1, GRID_SIZE):
        line = [[0, y], [width, y]]
        yLines.append(line)

    display, regions = draw(img.copy())

    while True:
        cv2.imshow('image', display)

        k = cv2.waitKey(10)

        if k == 27:    # Esc key to stop
            break
        elif k == 81:   # left
            display = shiftGrid(regions.copy(), 0, -SHIFT_SIZE)
        elif k == 83:   # right
            display = shiftGrid(regions.copy(), 0, SHIFT_SIZE)
        elif k == 84:   # up
            display = shiftGrid(regions.copy(), 1, SHIFT_SIZE)
        elif k == 82:   # down
            display = shiftGrid(regions.copy(), 1, -SHIFT_SIZE)
        elif k == 61:
            K += 8
            draw(img.copy())
        elif k == 45:
            K -= 8
            draw(img.copy())
        # else:
        #     if k != -1:
        #         print(k)

        # Reshow in case of updates
        cv2.imshow('image', display)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
