import cv2
import math
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.autograd import Variable
from advertorch.utils import predict_from_logits
from advertorch.attacks import JSMA

import interpolate
import inpaint
import other

from parse_attack_stat import *
from model import Cifar10CnnModel, Cifar10CnnModelGray, MnistModel, to_device


def normalize_full(a):
    a_min = a.min(axis=(0, 1), keepdims=True)
    a_max = a.max(axis=(0, 1), keepdims=True)
    return (a - a_min) / (a_max - a_min)


def mean(lst):
    res = 0

    for item in lst:
        res += item
    res /= len(lst)

    return res


def l2(list_1: list, list_2: list):
    if len(list_1) == len(list_2):
        dist = 0

        for i in range(len(list_1)):
            dist += (list_1[i] - list_2[i]) ** 2

        return math.sqrt(dist)


def l2(num_1: float, num_2: float):
    return abs(num_1 - num_2)


def length(lst):
    l = 0

    for item in lst:
        l += item ** 2

    return math.sqrt(l) if l > 0 else 1


def angle(first, second):
    try:
        if len(first) == len(second):
            cos = 0

            for i in range(len(first)):
                cos += first[i] * second[i]
            cos /= length(first)
            cos /= length(second)

            return cos if cos != 0 else 1
    except BaseException:
        return 1


def get_original_image(img, pixels, save=True, save_path=""):
    original_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for pixel in pixels:
        x = pixel['position'][0]
        y = pixel['position'][1]
        true_pixel = pixel['cln_color']

        original_img[x][y] = true_pixel

    if save:
        plt.imsave(save_path, original_img)

    return original_img


def restore_pixel(restore_mode, img, x, y):
    if restore_mode == 'bilinear':
        restored_pixel = interpolate.bilinear(img=img, x=x, y=y)
    elif restore_mode == 'bicubic':
        restored_pixel = interpolate.bicubic(img=img, x=x, y=y)
    elif restore_mode == 'lanczos':
        restored_pixel = interpolate.lanczos(img=img, x=x, y=y)
    elif restore_mode == 'telea':
        restored_pixel = inpaint.telea(img=img, x=x, y=y)
    elif restore_mode == 'navier_stokes':
        restored_pixel = inpaint.navier_stokes(img=img, x=x, y=y)
    elif restore_mode == 'gauss_blur':
        restored_pixel = other.gauss_blur(img=img, x=x, y=y)
    elif restore_mode == 'fourier_transform':
        restored_pixel = other.fourier_transform(img=img, x=x, y=y)
    else:
        raise Exception(f"Wrong restore mode: {restore_mode}")
    return restored_pixel[::-1]


def restore_pixel_gray(restore_mode, img, x, y):
    if restore_mode == 'bilinear':
        restored_pixel = interpolate.bilinear_gray(img=img, x=x, y=y)
    elif restore_mode == 'lanczos':
        restored_pixel = interpolate.lanczos_gray(img=img, x=x, y=y)
    elif restore_mode == 'telea':
        restored_pixel = inpaint.telea_gray(img=img, x=x, y=y)
    elif restore_mode == 'navier_stokes':
        restored_pixel = inpaint.navier_stokes_gray(img=img, x=x, y=y)
    elif restore_mode == 'gauss_blur':
        restored_pixel = other.gauss_blur(img=img, x=x, y=y)
    elif restore_mode == 'fourier_transform':
        restored_pixel = other.fourier_transform_gray(img=img, x=x, y=y)
    else:
        raise Exception(f"Wrong restore mode: {restore_mode}")
    return restored_pixel


def restore_image(restore_mode, img, pixels, save=True, save_path="", compute_metrics=True):
    distances = []
    metrics = []
    restored_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for pixel in pixels:
        x = pixel['position'][0]
        y = pixel['position'][1]

        restored_pixel = restore_pixel(restore_mode=restore_mode, img=img, x=x, y=y)

        if compute_metrics:
            true_pixel = pixel['cln_color']
            dist = l2(true_pixel, restored_pixel)
            distances.append(dist)

            cos = angle(true_pixel, restored_pixel)
            metric = dist / cos
            metrics.append(metric)

        img[x][y] = restored_pixel[::-1]
        restored_img[x][y] = restored_pixel

    if save:
        plt.imsave(save_path, restored_img)

    if compute_metrics:
        return restored_img, distances, metrics
    else:
        return restored_img


def restore_image_gray(restore_mode, img, pixels, save=True, save_path="", compute_metrics=True):
    distances = []
    metrics = []
    restored_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for pixel in pixels:
        x = pixel['position'][0]
        y = pixel['position'][1]

        true_pixel = pixel['cln_color']
        restored_pixel = restore_pixel_gray(restore_mode=restore_mode, img=img, x=x, y=y)

        if compute_metrics:
            dist = l2(true_pixel, restored_pixel)
            distances.append(dist)

            cos = angle(true_pixel, restored_pixel)
            metric = dist / cos
            metrics.append(metric)

        img[x][y] = restored_pixel[::-1]
        restored_img[x][y] = restored_pixel

        if save:
            plt.imsave(save_path, restored_img)

    if compute_metrics:
        return restored_img, distances, metrics
    else:
        return restored_img


# w = 70
# h = 70
# x_min = 31
# x_max = 101
# y_min = 208
# y_max = 278
# pixels = []
# for x in range(x_min, x_max + 1):
#     for y in range(y_min, y_max + 1):
#         pixels.append({'position': [x, y]})
#
# img = cv2.imread("ImageNet/adversarial.png")
# restore_image(restore_mode='lanczos', img=img,pixels=pixels, save=True, save_path="ImageNet/!l_restored.png", compute_metrics=False)
#
# raise BaseException("DONE!")

path = "CIFAR10GRAY/jsma 005/"
if 'jsma' in path:
    stat = parse_attack_stat_jsma(path + "!result.txt", mode='g')
else:
    stat = parse_attack_stat_1px(path + "!result.txt", mode='g')
path_attacked_images = path + "attacked images/"
path_restored_images = path + "restored images/"
path_original_images = path + "original images/"

restore_modes = []
restore_modes.append('bilinear')
restore_modes.append('lanczos')
restore_modes.append('telea')
restore_modes.append('navier_stokes')
restore_modes.append('gauss_blur')
restore_modes.append('fourier_transform')

best = 10 ** 8

# for key in stat.keys():
#     img = cv2.imread(path_attacked_images + key)
#     get_original_image(img, stat[key]['pixels'], save=True, save_path=path_original_images + key)

header = "method\tdist\tcos\tangle\tmetric"
output = open(path + '/' + '!_' + path.split('/')[1] + '.txt', 'w')
output.write(header + '\n')
output.close()
print(header)

time_header = "method\ttime"
output_time = open('!g_' + path.split('/')[1] + '_time.txt', 'w')
output_time.write(time_header + '\n')
output_time.close()

for restore_mode in restore_modes:
    dists = []
    coses = []
    angles = []
    metrics = []
    time_compute = 0
    for key in stat.keys():
        img = cv2.imread(path_attacked_images + key)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        restored_img = img # cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for pixel in stat[key]['pixels']:
            x = pixel['position'][0]
            y = pixel['position'][1]

            true_pixel = pixel['cln_color']
            time_start = time.time()
            restored_pixel = restore_pixel_gray(restore_mode=restore_mode, img=img, x=x, y=y)
            time_compute += time.time() - time_start

            dist = l2(true_pixel, restored_pixel)
            cos = angle(true_pixel, restored_pixel)
            cos = cos if cos <= 1 else 1
            cos = cos if cos >= 0 else 0
            ang = math.acos(cos) * 180.0 / math.pi
            try:
                metric = dist / cos
            except ZeroDivisionError:
                metric = math.inf

            dists.append(dist)
            coses.append(cos)
            angles.append(ang)
            metrics.append(metric)

            img[x][y] = restored_pixel# [::-1]
            restored_img[x][y] = restored_pixel

        # plt.imsave(f"{path_restored_images}{restore_mode}/{key}", restored_img)
        img = (normalize_full(restored_img) * 255).astype(np.uint8)
        img = Image.fromarray(img)
        img.save(f"{path_restored_images}{restore_mode}/{key}")

    mean_dist = mean(dists)
    mean_cos = mean(coses)
    mean_ang = mean(angles)
    mean_metric = mean(metrics)

    line = f"{restore_mode}\t{mean_dist}\t{mean_cos}\t{mean_ang}\t{mean_metric}"
    output = open('!g_' + path.split('/')[1] + '.txt', 'a')
    output.write(line + '\n')
    output.close()
    print(line)

    time_line = f"{restore_mode}\t{time_compute}"
    output_time = open('!g_' + path.split('/')[1] + '_time.txt', 'a')
    output_time.write(time_line + '\n')
    output_time.close()
    print(time_line)

    print()

    if mean_dist < best:
        best = mean_dist

print(best)
raise BaseException("DONE!")

torch.manual_seed(42)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

model = to_device(MnistModel(), device)
model.load_state_dict(torch.load('models/mnist-cnn.pth'))
# model = to_device(Cifar10CnnModelGray(), device)
# model.load_state_dict(torch.load('models/cifar10_gray-cnn.pth'))
# model = to_device(Cifar10CnnModel(), device)
# model.load_state_dict(torch.load('models/cifar10-cnn.pth'))

correction_stat = {}
for rm in restore_modes:
    correction_stat[rm] = 0


def predict(network, input_path):
    img = cv2.imread(input_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)
    i = img.tolist()
    ii = [i]
    img = torch.FloatTensor(ii).to(device)
    p_label = network(img).tolist()[0]
    return p_label.index(max(p_label))


def predict_gray(network, input_path):
    img = cv2.imread(input_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = img.transpose(2, 0, 1)
    i = img.tolist()
    ii = [[i]]
    img = torch.FloatTensor(ii).to(device)
    p_label = network(img).tolist()[0]
    return p_label.index(max(p_label))


ttl_cnt = 0

for key in stat.keys():
    c = stat[key]['cln_label']
    a = stat[key]['adv_label']
    # cln_label = predict(network=model, input_path=f"{path_original_images}{key}")
    # adv_label = predict(network=model, input_path=f"{path_attacked_images}{key}")
    cln_label = predict_gray(network=model, input_path=f"{path_original_images}{key}")
    adv_label = predict_gray(network=model, input_path=f"{path_attacked_images}{key}")
    if cln_label != adv_label:
        ttl_cnt += 1
        for rm in restore_modes:
            img = cv2.imread(path_restored_images + rm + '/' + key)
            # p_label = predict(network=model, input_path=f"{path_restored_images}{rm}/{key}")
            p_label = predict_gray(network=model, input_path=f"{path_restored_images}{rm}/{key}")

            if p_label == cln_label:
                correction_stat[rm] += 1

print(f"{len(stat.keys())}\t\t\t{ttl_cnt}")

output = open('!bw_' + path.split('/')[1] + '_fix' + '.txt', 'a')
# output = open('!bw_' + path.split('/')[1] + '_fix_full' + '.txt', 'a')

for key in correction_stat.keys():
    correction_stat[key] /= ttl_cnt
    s = f"{key}\t{correction_stat[key]}"
    print(s)
    output.write(s + '\n')

output.write(f"\n\n{len(stat.keys())}\t{ttl_cnt}")
output.close()
