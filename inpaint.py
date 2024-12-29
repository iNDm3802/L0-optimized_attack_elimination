import cv2
import numpy as np


def navier_stokes(img, x: int, y: int, mode='c'):
    # change to img shape
    mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    mask[x, y] = 1
    if mode == 'c':
        inpaint = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)
    else:
        inpaint = cv2.inpaint(img, mask, 1, cv2.INPAINT_NS)

    if mode == 'c':
        return inpaint[x][y][::-1]
    else:
        return inpaint[x][y]


def telea(img, x: int, y: int, mode='c'):
    # change to img shape
    mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    mask[x, y] = 1
    if mode == 'c':
        inpaint = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    else:
        inpaint = cv2.inpaint(img, mask, 1, cv2.INPAINT_TELEA)

    if mode == 'c':
        return inpaint[x][y][::-1]
    else:
        return inpaint[x][y]
