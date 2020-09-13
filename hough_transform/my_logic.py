import numpy as np
import cv2


def hough_line(img, edge):
    # Hough Transform
    # Theta 0 - 180 degree
    # Calculate 'cos' and 'sin' value ahead to improve running time
    thetas = np.arange(0, 180, 1)
    cos = np.cos(np.deg2rad(thetas))  # convert the angle to radian
    sin = np.sin(np.deg2rad(thetas))

    # Generate a accumulator matrix the Hough Space with dimension to store the values
    width, height = img.shape[0], img.shape[1]
    diag_len = int(round((width ** 2 + height ** 2) ** 0.5))
    accumulator = np.zeros((2 * diag_len, len(thetas)), dtype=np.uint8)

    # rho: Distance resolution of the accumulator in pixels.
    # theta: Angle resolution of the accumulator in radians.(np.pi / 180)
    # *threshold: Accumulator threshold, line selection.
    # Threshold to get edges pixel location (x,y)
    edge_pixels = np.where(edge == 255)
    coordinates = list(zip(edge_pixels[0], edge_pixels[1]))

    # Calculate rho value for each edge location (x,y) with all the theta range and then add it to the accumulator.
    for p in range(len(coordinates)):
        for t in range(len(thetas)):
            rho = int(round(coordinates[p][1] * cos[t] + coordinates[p][0] * sin[t]))
            accumulator[rho, t] += 2
    return accumulator


def lines_by_convolution(img):
    # horizontal
    r1 = np.array([[-1, -1, -1],
                   [2, 2, 2],
                   [-1, -1, -1]], dtype=np.float64)

    # vertical
    r2 = np.array([[-1, 2, -1],
                   [-1, 2, -1],
                   [-1, 2, -1]], dtype=np.float64)
    # 45 diagonal
    r3 = np.array([[-1, -1, 2],
                   [-1, 2, -1],
                   [2, -1, -1]], dtype=np.float64)
    # -45 diagonal
    r4 = np.array([[2, -1, -1],
                   [-1, 2, -1],
                   [-1, -1, 2]], dtype=np.float64)

    # convolved pictures
    r11 = cv2.filter2D(img, -1, r1)
    r21 = cv2.filter2D(img, -1, r2)
    #r31 = cv2.filter2D(img, -1, r3)
    #r41 = cv2.filter2D(img, -1, r4)
    # find maximum between two of them 
    rmax = np.maximum(r11, r21)
    result = np.zeros(r11.shape)
    result[rmax == 255] = 255

    return result

