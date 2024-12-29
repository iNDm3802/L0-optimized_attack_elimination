import cv2
import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from interpolate import sum_by_index


def neighbors(img, x: int, y: int, mode='c'):
    x_left = x - 1 if x > 0 else False
    x_right = x + 1 if x < img.shape[0] - 1 else False
    y_bottom = y - 1 if y > 0 else False
    y_top = y + 1 if y < img.shape[1] - 1 else False

    neighbors = []
    if type(x_left) != type(False):
        neighbors.append(img[x_left][y])
    if type(x_right) != type(False):
        neighbors.append(img[x_right][y])
    if type(y_bottom) != type(False):
        neighbors.append(img[x][y_bottom])
    if type(y_top) != type(False):
        neighbors.append(img[x][y_top])
    if type(x_left) != type(False) and type(y_top) != type(False):
        neighbors.append(img[x_left][y_top])
    if type(x_right) != type(False) and type(y_top) != type(False):
        neighbors.append(img[x_right][y_top])
    if type(x_left) != type(False) and type(y_bottom) != type(False):
        neighbors.append(img[x_left][y_bottom])
    if type(x_right) != type(False) and type(y_bottom) != type(False):
        neighbors.append(img[x + 1][y_bottom])

    if mode == 'c':
        restored_px_r = int(sum_by_index(neighbors, 2) / len(neighbors))
        restored_px_g = int(sum_by_index(neighbors, 1) / len(neighbors))
        restored_px_b = int(sum_by_index(neighbors, 0) / len(neighbors))
        restored_px = [restored_px_b, restored_px_g, restored_px_r]
    else:
        restored_px = int(sum(neighbors) / len(neighbors))

    return restored_px


def gauss_blur(img, x: int, y: int, mode='c'):

    filtered_img = cv2.GaussianBlur(img, (3, 3), 2.0)
    if mode == 'c':
        return filtered_img[x][y][::-1]
    else:
        return filtered_img[x][y]


def restore_channel(channel, corrupted_coords):
    dft_channel = fft2(channel)
    dft_channel_shifted = fftshift(dft_channel)

    center = np.array(dft_channel_shifted.shape) // 2
    dft_corrupted_coords = (center[0] - corrupted_coords[0], center[1] - corrupted_coords[1])

    dft_channel_shifted[dft_corrupted_coords] = 0
    dft_channel = ifftshift(dft_channel_shifted)

    restored_channel = np.abs(ifft2(dft_channel))
    return restored_channel


def fourier_transform(img, x: int, y: int, mode='c'):
    if mode == 'c':
        corrupted_b, corrupted_g, corrupted_r = cv2.split(img)

        restored_b = restore_channel(corrupted_b, (x, y))
        restored_g = restore_channel(corrupted_g, (x, y))
        restored_r = restore_channel(corrupted_r, (x, y))

        restored_image = cv2.merge((restored_b, restored_g, restored_r)).astype(np.uint8)
    else:
        restored = restore_channel(img, (x, y))

        restored_image = restored.astype(np.uint8)

    return restored_image[x][y]