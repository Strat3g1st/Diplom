import os
from PIL import Image
import numpy as np
import cv2


def format_to_256x256(path_old, path_new):
    img = Image.open(path_old)
    width, height = img.size
    if width > height:
        left = int((width - height)/2)
        top = 0
        right = width - left
        bottom = height
        img = img.crop((left, top, right, bottom))
    if width < height:
        left = 0
        top = int((height - width) / 2)
        bottom = height - top
        right = width
        img = img.crop((left, top, right, bottom))
    img = img.resize((256, 256), Image.ANTIALIAS)
    img.save(path_new)


path_old = "C:\\Users\\maste\\Documents\\objects\\"
path_new = "C:\\Users\\maste\\Documents\\objects\\"

mantis = "mantis\\"
mapple = "mapple\\"
vinca = "vinca\\"
wasp = "wasp\\"

L = [mantis, mapple, vinca, wasp]

images = "images\\"
masks = "masks\\"

DFN = "DFN\\"
CRA = "CRA\\"
HII = "HII\\"


def create_dir():
    os.mkdir(path_new)
    for object in L:
        os.mkdir(path_new + object)
        os.mkdir(path_new + object + images)
        os.mkdir(path_new + object + masks)
        os.mkdir(path_new + object + DFN)
        os.mkdir(path_new + object + CRA)
        os.mkdir(path_new + object + HII)


def create_db():
    for object in L:
        image_names = os.listdir(path_old + object + images)
        for name in image_names:
            print(path_old + object + images + name)
            format_to_256x256(path_old + object + images + name, path_new + object + images + name)
        mask_names = os.listdir(path_old + object + masks)
        for name in mask_names:
            print(path_old + object + masks + name)
            format_to_256x256(path_old + object + masks + name, path_new + object + masks + name)


def format_mask(path_old, path_new):
    img = Image.open(path_old)
    img = img.convert('RGB')
    img = np.asarray(img)
    black = [0, 0, 0]
    white = [255, 255, 255]
    r_img, g_img, b_img = img[:, :, 0].copy(), img[:, :, 1].copy(), img[:, :, 2].copy()
    img = r_img + g_img + b_img
    backgorund = black[0] + black[1] + black[2]
    r_img[img == backgorund] = white[0]
    g_img[img == backgorund] = white[1]
    b_img[img == backgorund] = white[2]
    r_img[img != backgorund] = black[0]
    g_img[img != backgorund] = black[1]
    b_img[img != backgorund] = black[2]
    result = np.dstack([r_img, g_img, b_img])
    result = Image.fromarray(result.astype(np.uint8))
    result.save(path_new)
    image = cv2.imread(path_new)
    processed_image = cv2.medianBlur(image, 3)
    cv2.imwrite(path_new, processed_image)


def apply_format_to_all():
    for object in L:
        mask_names = os.listdir(path_old + object + masks)
        for name in mask_names:
            format_mask(path_new + object + masks + name, path_new + object + masks + name)
            print(path_new + object + masks + name)


# create_dir()
create_db()
apply_format_to_all()

