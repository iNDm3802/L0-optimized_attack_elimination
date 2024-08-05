import cv2
import numpy as np


def navier_stokes(img, x: int, y: int):
    # change to img shape
    mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    mask[x, y] = 1
    inpaint = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)

    return inpaint[x][y]


def telea(img, x: int, y: int):
    # change to img shape
    mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    mask[x, y] = 1
    inpaint = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

    return inpaint[x][y]


def navier_stokes_gray(img, x: int, y: int):
    # change to img shape
    mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    mask[x, y] = 1
    inpaint = cv2.inpaint(img, mask, 1, cv2.INPAINT_NS)

    return inpaint[x][y]


def telea_gray(img, x: int, y: int):
    # change to img shape
    mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    mask[x, y] = 1
    inpaint = cv2.inpaint(img, mask, 1, cv2.INPAINT_TELEA)

    return inpaint[x][y]