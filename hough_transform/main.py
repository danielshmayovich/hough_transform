from matplotlib import pyplot as plt
import numpy as np
import cv2
import my_logic as ml

img = cv2.imread('square.jpeg')
# convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# find the edges by Canny Edge Detection
edges = cv2.Canny(gray, 10, 50)

# Function to do hough line transform
accumulator = ml.hough_line(img, edges)

# Threshold to take matrix indexes location.
# coordinates[i][0] = rho in i position, coordinates[i][1] = theta in i position.
edge_pixels = np.where(accumulator > 150)
coordinates = list(zip(edge_pixels[0], edge_pixels[1]))


# Probabilistic Hough Transform
minLineLength = 100
maxLineGap = 10
# HoughLinesP need to get picture, rho size, theta, threshold, minimum line length, max line gap
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength, maxLineGap)
img_cpy2 = img.copy()
# Draw all the line on the image - houghlinep returns dots on a the line
for i in range(0, len(lines)):
    for x1, y1, x2, y2 in lines[i]:
        cv2.line(img_cpy2, (x1, y1), (x2, y2), (0, 255, 0), 2)



img_cpy = img.copy()
# Use line equation to draw detected line on an original image
for i in np.arange(len(coordinates)):
    a = np.cos(np.deg2rad(coordinates[i][1]))
    b = np.sin(np.deg2rad(coordinates[i][1]))
    if a == 0:
        a = 1
    if b == 0:
        b = 1

    x0 = coordinates[i][0] * a
    y0 = coordinates[i][0] * b
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * a)
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * a)
    cv2.line(img_cpy, (x1, y1), (x2, y2), (0, 255, 0), 1)

conv_lines = ml.lines_by_convolution(img.copy())

# show result
plt.figure()
plt.subplot(231)
plt.title('original')
plt.imshow(img[:, :, ::-1])
plt.subplot(232)
plt.title('edges map')
plt.imshow(edges, cmap='gray')
plt.subplot(233)
plt.title('with lines')
plt.imshow(img_cpy[:, :, ::-1])
plt.subplot(234)
plt.title("Hough Space")
plt.imshow(accumulator, cmap='jet')
plt.subplot(235)
plt.title('Probabilistic Hough Transform')
plt.imshow(img_cpy2[:, :, ::-1])
plt.subplot(236)
plt.title('finding lines by convolution ')
plt.imshow(conv_lines, cmap='gray')
plt.show()
