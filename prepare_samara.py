import os
from PIL import Image


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


def apply_format_to_all(path):
    names = os.listdir(path)
    for name in names:
        print(path + name)
        format_to_256x256(path + name, path + name.split('.')[0] + '.png')


path = "C:\\Users\\maste\\Documents\\dataset_samara_backup\\images\\"
apply_format_to_all(path)