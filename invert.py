import os
from PIL import Image
import numpy as np


def img_open(path):
    img = Image.open(path)
    img = img.convert('RGB')
    img = np.asarray(img)
    return img


def img_save(img, path):
    result = Image.fromarray(img.astype(np.uint8))
    result.save(path)


def invert_img(img):
    new_img = img.copy()
    new_img[img == 255] = 0
    new_img[img == 0] = 255
    return new_img


def norm_img(img):
    new_img = img.copy()
    new_img[img >= 127] = 255
    new_img[img < 127] = 0
    return new_img


def apply_invert_to_all(path_mask):
    L = os.listdir(path_mask)
    for l in L:
        path = path_mask + l
        print(path)
        img = img_open(path)
        img = norm_img(img)
        img = invert_img(img)
        img_save(img, os.path.splitext(path)[0] + '.png')


name_dir_data = "objects"
path1 = "C:\\Users\\maste\\Documents\\" + name_dir_data + "\\mantis\\masks\\"
path2 = "C:\\Users\\maste\\Documents\\" + name_dir_data + "\\mapple\\masks\\"
path3 = "C:\\Users\\maste\\Documents\\" + name_dir_data + "\\vinca\\masks\\"
path4 = "C:\\Users\\maste\\Documents\\" + name_dir_data + "\\wasp\\masks\\"


apply_invert_to_all(path1)
apply_invert_to_all(path2)
apply_invert_to_all(path3)
apply_invert_to_all(path4)
