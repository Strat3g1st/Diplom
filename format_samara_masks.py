import os
from PIL import Image
import numpy as np
import cv2


def format_mask(path_old, path_new):
    img = Image.open(path_old)
    img = img.convert('RGB')
    img = np.asarray(img)
    img = img[:, :, 0]
    result = img.copy()
    result[img > 10] = 255
    result = Image.fromarray(result.astype(np.uint8))
    result.save(path_new)
    image = cv2.imread(path_new)
    processed_image = cv2.medianBlur(image, 3)
    processed_image = cv2.medianBlur(processed_image, 3)
    processed_image = cv2.medianBlur(processed_image, 3)
    cv2.imwrite(path_new, processed_image)


def apply_format_to_all(path):
    names = os.listdir(path)
    for name in names:
        print(path + name)
        format_mask(path + name, path + name.split('.')[0] + '.png')


path = "C:\\Users\\maste\\Documents\\dataset_samara\\masks\\"
format_mask(path + '37.png', path + '37.png')
# apply_format_to_all(path)
