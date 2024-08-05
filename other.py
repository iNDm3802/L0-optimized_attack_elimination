import cv2
import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift


def restore_channel(channel, corrupted_coords):
    dft_channel = fft2(channel)
    dft_channel_shifted = fftshift(dft_channel)

    center = np.array(dft_channel_shifted.shape) // 2
    dft_corrupted_coords = (center[0] - corrupted_coords[0], center[1] - corrupted_coords[1])

    dft_channel_shifted[dft_corrupted_coords] = 0
    dft_channel = ifftshift(dft_channel_shifted)

    restored_channel = np.abs(ifft2(dft_channel))
    return restored_channel


def gauss_blur(img, x: int, y: int):

    filtered_img = cv2.GaussianBlur(img, (3, 3), 2.0)
    return filtered_img[x][y]


def fourier_transform(img, x: int, y: int):
    corrupted_b, corrupted_g, corrupted_r = cv2.split(img)

    restored_b = restore_channel(corrupted_b, (x, y))
    restored_g = restore_channel(corrupted_g, (x, y))
    restored_r = restore_channel(corrupted_r, (x, y))

    restored_image = cv2.merge((restored_b, restored_g, restored_r)).astype(np.uint8)
    return restored_image[x][y]


def fourier_transform_gray(img, x: int, y: int):
    restored = restore_channel(img, (x, y))

    restored_image = restored.astype(np.uint8)
    return restored_image[x][y]