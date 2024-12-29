import numpy as np


def sum_by_index(lst, index):
    res = 0
    res = float(res)
    for item in lst:
        res += item[index]
    return res


def bilinear(img, x: int, y: int, mode='c'):
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

    if mode == 'c':
        restored_px_r = int(sum_by_index(neighbors, 2) / len(neighbors))
        restored_px_g = int(sum_by_index(neighbors, 1) / len(neighbors))
        restored_px_b = int(sum_by_index(neighbors, 0) / len(neighbors))
        restored_px = [restored_px_b, restored_px_g, restored_px_r]
    else:
        restored_px = int(sum(neighbors) / len(neighbors))

    return restored_px


def bicubic(img, x: int, y: int, mode='c'):
    def cubic_convolution_coefficient(xx):
        abs_x = np.abs(xx)
        if abs_x <= 1:
            return 1 - 2 * abs_x ** 2 + abs_x ** 3
        elif abs_x <= 2:
            return 4 - 8 * abs_x + 5 * abs_x ** 2 - abs_x ** 3
        else:
            return 0

    original_array = np.copy(img)
    original_array[x][y] = bilinear(img, x, y, mode=mode)

    x_left = x - 1 if x >= 1 else False
    x_left_2 = x - 2 if x >= 2 else False
    x_right = x + 1 if x < img.shape[0] - 1 else False
    x_right_2 = x + 2 if x < img.shape[0] - 2 else False
    y_bottom = y - 1 if y >= 1 else False
    y_bottom_2 = y - 2 if y >= 2 else False
    y_top = y + 1 if y < img.shape[1] - 1 else False
    y_top_2 = y + 2 if y < img.shape[1] - 2 else False

    row_indices = []
    if type(x_left_2) != type(False):
        row_indices.append(x_left_2)
    if type(x_left) != type(False):
        row_indices.append(x_left)
    if type(x_right) != type(False):
        row_indices.append(x_right)
    if type(x_right_2) != type(False):
        row_indices.append(x_right_2)

    col_indices = []
    if type(y_bottom_2) != type(False):
        col_indices.append(y_bottom_2)
    if type(y_bottom) != type(False):
        col_indices.append(y_bottom)
    if type(y_top) != type(False):
        col_indices.append(y_top)
    if type(y_top_2) != type(False):
        col_indices.append(y_top_2)

    if mode == 'c':
        Z_red = original_array[row_indices][:, col_indices, 0]
        Z_green = original_array[row_indices][:, col_indices, 1]
        Z_blue = original_array[row_indices][:, col_indices, 2]

        weights = np.zeros((len(row_indices), len(col_indices)))
        for i in range(len(row_indices)):
            for j in range(len(col_indices)):
                x_dist = (row_indices[i] - x) / len(row_indices)
                y_dist = (col_indices[j] - y) / len(col_indices)
                weights[i, j] = cubic_convolution_coefficient(x_dist) * cubic_convolution_coefficient(y_dist)

        weights = weights.reshape((-1, 1))
        Z_red = Z_red.flatten()
        Z_green = Z_green.flatten()
        Z_blue = Z_blue.flatten()

        Z2_red = np.dot(weights.T, Z_red)
        Z2_green = np.dot(weights.T, Z_green)
        Z2_blue = np.dot(weights.T, Z_blue)

        divider = sum(weights)
        restored_px = [int(Z2_blue / divider), int(Z2_green / divider), int(Z2_red / divider)]
    else:
        Z = original_array[row_indices][:, col_indices]
        weights = np.zeros((len(row_indices), len(col_indices)))
        for i in range(len(row_indices)):
            for j in range(len(col_indices)):
                x_dist = (row_indices[i] - x) / len(row_indices)
                y_dist = (col_indices[j] - y) / len(col_indices)
                weights[i, j] = cubic_convolution_coefficient(x_dist) * cubic_convolution_coefficient(y_dist)
        weights = weights.reshape((-1, 1))
        Z = Z.flatten()

        Z2 = np.dot(weights.T, Z)

        divider = sum(weights)
        restored_px = int(Z2) / divider

    return restored_px[0]


def lanczos(img, x: int, y: int, window_size=3, a=2, mode='c'):
    def lanczos_kernel(xx):
        if xx == 0:
            return 1
        elif abs(xx) < window_size:
            # return aa * np.sin(np.pi * xx) * np.sin(np.pi * xx / aa) / (np.pi ** 2 * xx ** 2)
            return a * np.sinc(xx) * np.sinc(xx / a)
        else:
            return 0

    original_array = np.copy(img)
    original_array[x][y] = bilinear(original_array, x, y, mode=mode)
    xmin = int(x) - window_size if int(x) - window_size > 0 else 0
    xmax = int(x) + window_size + 1 if int(x) + window_size + 1 < img.shape[0] - 1 else img.shape[0] - 1
    ymin = int(y) - window_size if int(y) - window_size > 0 else 0
    ymax = int(y) + window_size + 1 if int(y) + window_size + 1 < img.shape[0] - 1 else img.shape[0] - 1

    numerator = 0
    denominator = 0
    if mode == 'c':
        rgb = [2, 1, 0]
        interpolated_pixel = [0, 0, 0]
        for chanel in rgb:
            for i in range(xmin, xmax):
                for j in range(ymin, ymax):
                    wx = lanczos_kernel(x - i)
                    wy = lanczos_kernel(y - j)
                    w = wx * wy
                    if 0 <= i < original_array.shape[0] and 0 <= j < original_array.shape[1]:
                        numerator += w * original_array[i][j][chanel]
                        denominator += w

            interpolated_pixel[chanel] = numerator / denominator

        return [round(res, 3) for res in interpolated_pixel]
    else:
        for i in range(xmin, xmax):
            for j in range(ymin, ymax):
                wx = lanczos_kernel(x - i)
                wy = lanczos_kernel(y - j)
                w = wx * wy

                if 0 <= i < original_array.shape[0] and 0 <= j < original_array.shape[1]:
                    numerator += w * original_array[i][j]
                    denominator += w

        interpolated_pixel = numerator / denominator

        return round(interpolated_pixel, 3)
