import numpy as np
from functools import reduce


def sum_by_index(lst, index):
    res = 0
    for item in lst:
        res += item[index]
    return res


def bilinear(img, x: int, y: int) -> list:
    x_left = x - 1 if x > 0 else False
    x_right = x + 1 if x < img.shape[0] - 1 else False
    y_bottom = y - 1 if y > 0 else False
    y_top = y + 1 if y < img.shape[1] - 1 else False

    neighbors = []
    if x_left and y_top:
        neighbors.append(img[x_left][y_top])
    if x_right and y_top:
        neighbors.append(img[x_right][y_top])
    if x_left and y_bottom:
        neighbors.append(img[x_left][y_bottom])
    if x_right and y_bottom:
        neighbors.append(img[x + 1][y_bottom])

    restored_px_r = int(sum_by_index(neighbors, 2) / len(neighbors))
    restored_px_g = int(sum_by_index(neighbors, 1) / len(neighbors))
    restored_px_b = int(sum_by_index(neighbors, 0) / len(neighbors))
    restored_px = [restored_px_b, restored_px_g, restored_px_r]

    return restored_px


def lanczos(img, x: int, y: int, window_size=3, a=2):
    def lanczos_kernel(xx):
        if xx == 0:
            return 1
        elif abs(xx) < window_size:
            # return aa * np.sin(np.pi * xx) * np.sin(np.pi * xx / aa) / (np.pi ** 2 * xx ** 2)
            return a * np.sinc(xx) * np.sinc(xx / a)
        else:
            return 0

    original_array = np.copy(img)
    original_array[x][y] = bilinear(original_array, x, y)
    # Compute the range of pixels to consider for interpolation
    xmin = int(x) - window_size if int(x) - window_size > 0 else 0
    xmax = int(x) + window_size + 1 if int(x) + window_size + 1 < img.shape[0] - 1 else img.shape[0] - 1
    ymin = int(y) - window_size if int(y) - window_size > 0 else 0
    ymax = int(y) + window_size + 1 if int(y) + window_size + 1 < img.shape[0] - 1 else img.shape[0] - 1

    # Compute the numerator and denominator of the interpolation equation
    numerator = 0
    denominator = 0
    rgb = [2, 1, 0]
    interpolated_pixel = [0, 0, 0]
    for chanel in rgb:
        for i in range(xmin, xmax):
            for j in range(ymin, ymax):
                # Compute the weight for the current pixel
                wx = lanczos_kernel(x - i)
                wy = lanczos_kernel(y - j)
                w = wx * wy
                # Add the weighted pixel value to the numerator and denominator
                if 0 <= i < original_array.shape[0] and 0 <= j < original_array.shape[1]:
                    numerator += w * original_array[i][j][chanel]
                    denominator += w

        # Compute the interpolated pixel value
        interpolated_pixel[chanel] = numerator / denominator

    return [round(res, 3) for res in interpolated_pixel]


def bilinear_gray(img, x: int, y: int) -> int:
    x_left = x - 1 if x > 0 else False
    x_right = x + 1 if x < img.shape[0] - 1 else False
    y_bottom = y - 1 if y > 0 else False
    y_top = y + 1 if y < img.shape[1] - 1 else False

    neighbors = []
    if x_left and y_top:
        neighbors.append(img[x_left][y_top])
    if x_right and y_top:
        neighbors.append(img[x_right][y_top])
    if x_left and y_bottom:
        neighbors.append(img[x_left][y_bottom])
    if x_right and y_bottom:
        neighbors.append(img[x + 1][y_bottom])

    restored_px = int(sum(neighbors) / len(neighbors))

    return restored_px


def lanczos_gray(img, x: int, y: int, window_size=3, a=2):
    def lanczos_kernel(xx):
        if xx == 0:
            return 1
        elif abs(xx) < window_size:
            # return aa * np.sin(np.pi * xx) * np.sin(np.pi * xx / aa) / (np.pi ** 2 * xx ** 2)
            return a * np.sinc(xx) * np.sinc(xx / a)
        else:
            return 0

    original_array = np.copy(img)
    original_array[x][y] = bilinear_gray(original_array, x, y)
    # Compute the range of pixels to consider for interpolation
    xmin = int(x) - window_size if int(x) - window_size > 0 else 0
    xmax = int(x) + window_size + 1 if int(x) + window_size + 1 < img.shape[0] - 1 else img.shape[0] - 1
    ymin = int(y) - window_size if int(y) - window_size > 0 else 0
    ymax = int(y) + window_size + 1 if int(y) + window_size + 1 < img.shape[0] - 1 else img.shape[0] - 1

    # Compute the numerator and denominator of the interpolation equation
    numerator = 0
    denominator = 0
    rgb = [2, 1, 0]
    interpolated_pixel = [0, 0, 0]
    for i in range(xmin, xmax):
        for j in range(ymin, ymax):
            # Compute the weight for the current pixel
            wx = lanczos_kernel(x - i)
            wy = lanczos_kernel(y - j)
            w = wx * wy
            # Add the weighted pixel value to the numerator and denominator
            if 0 <= i < original_array.shape[0] and 0 <= j < original_array.shape[1]:
                numerator += w * original_array[i][j]
                denominator += w

    # Compute the interpolated pixel value
    interpolated_pixel = numerator / denominator

    return round(interpolated_pixel, 3)
