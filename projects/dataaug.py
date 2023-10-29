import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import albumentations as albu
import argparse

def contrast_image_correction(src):
    h, w, _ = src.shape
    src = cv2.resize(src, (w, h))
    old_y = cv2.cvtColor(src, cv2.COLOR_BGRA2YUV_I420)
    temp = cv2.bilateralFilter(old_y, 9, 50, 50)
    dst = np.zeros((h, w, 3))
    src = src.astype(np.int16)
    old_y = old_y.astype(np.int16)
    temp = temp.astype(np.int16)
    dst = dst.astype(np.int16)
    for i in range(h):
        for j in range(w):
            exp = 2 ** ((128 - (255 - temp[i][j])) / 128.0)
            value = int(255 * ((old_y[i][j] / 255.0) ** exp))
            temp[i][j] = value
    for i in range(h):
        for j in range(w):
            if old_y[i][j] == 0:
                for k in range(3):
                    dst[i][j][k] = 0
            else:
                dst[i][j][0] = (int(temp[i][j])*(src[i][j][0] + old_y[i][j])/(old_y[i][j]) + (src[i][j][0]) - old_y[i][j])/2
                dst[i][j][1] = (int(temp[i][j])*(src[i][j][1] + old_y[i][j])/(old_y[i][j]) + (src[i][j][1]) - old_y[i][j])/2
                dst[i][j][2] = (int(temp[i][j])*(src[i][j][2] + old_y[i][j])/(old_y[i][j]) + (src[i][j][2]) - old_y[i][j])/2
    dst = dst.astype(np.uint8)
    return dst

parser = argparse.ArgumentParser(description='Process some images.')

parser.add_argument('data_base_dir', type=str, help='input images path')
parser.add_argument('outfile_contrast', type=str, help='path of modified contrast of images')
parser.add_argument('results', type=str, help='path of sharpened images after modified contrast')

args = parser.parse_args()

if not os.path.exists(args.outfile_contrast):
    os.makedirs(args.outfile_contrast)
if not os.path.exists(args.results):
    os.makedirs(args.results)

for file in os.listdir(args.data_base_dir):
    read_img_name = os.path.join(args.data_base_dir, file.strip())
    image = cv2.imread(read_img_name)
    out_img_name = os.path.join(args.outfile_contrast, file.strip())
    hori, wori = image.shape[0], image.shape[1]
    h, w = image.shape[0] // 2 * 2, image.shape[1] // 2 * 2
    image = image[0:h, 0:w]
    image_correct = contrast_image_correction(image)
    dst = cv2.resize(image_correct, (wori, hori))
    cv2.imwrite(out_img_name, image_correct)

file_path = os.listdir(args.outfile_contrast)

for i in file_path:
    image = cv2.imread(os.path.join(args.outfile_contrast, i))
    aug = albu.Sharpen(alpha=0.2, lightness=0.5, p=1)
    img_HorizontalFlip = aug(image=image)['image']
    cv2.imwrite(os.path.join(args.results, i), img_HorizontalFlip)
