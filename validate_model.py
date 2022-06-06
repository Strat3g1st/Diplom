import os
import torch
from UNet256 import CNNNet
from PIL import Image
import numpy as np


def invert(img):
    img[img < 127] = 1
    img[img >= 127] = 2
    img[img == 1] = 255
    img[img == 2] = 0
    return img


def norm_img(img):
    new_img = img.copy()
    new_img[img >= 127] = 255
    new_img[img < 127] = 0
    return new_img


def img_to_bool(img):
    new_img = img.copy()
    new_img[img == 255] = 1
    return np.array(new_img, dtype=bool)


def iou_numpy(output: np.array, label: np.array):
    overlap = output * label  # Logical AND
    union = output + label  # Logical OR
    return overlap.sum() / float(union.sum())


def detect(model, path_image, path_mask):
    test_mask = Image.open(path_mask)
    test_mask = test_mask.convert('RGB')
    test_mask = np.asarray(test_mask)
    test_mask = test_mask[:, :, 0]
    test_mask = norm_img(test_mask)
    bool_mask = img_to_bool(test_mask)

    test_img = Image.open(path_image)
    test_img = test_img.convert('RGB')
    test_img = np.asarray(test_img)
    test_img = np.expand_dims(test_img, axis=0)

    test_img = torch.from_numpy(test_img)
    # test_img.cuda()
    test_img = test_img.type(torch.cuda.FloatTensor)
    test_img = test_img.reshape(test_img.shape[0], test_img.shape[3], test_img.shape[1], test_img.shape[2])

    result_mask = model.forward(test_img)
    result_mask = torch.max(result_mask, 1)[0]
    result_mask = result_mask.cpu().detach().numpy()
    result_mask = result_mask.astype(np.uint8)
    result_mask = np.squeeze(result_mask, axis=0)
    result_mask = norm_img(result_mask)
    result_mask = invert(result_mask)
    bool_result = img_to_bool(result_mask)

    return iou_numpy(bool_result, bool_mask)


losses = []
good = []


def validate(model_name, path_to_images, path_to_masks, min_iou):
    torch.cuda.device('cuda')
    torch.cuda.empty_cache()
    model = CNNNet()
    model.load_state_dict(torch.load(model_name))
    model.eval()
    model.cuda()
    for i in range(len(path_to_images)):
        iou = detect(model, path_to_images[i], path_to_masks[i])
        print(str(i + 1) + '/' + str(len(path_to_images)) + ': ' + str(iou))
        if iou >= min_iou:
            good.append(iou)


def sort_special(method, lst, len):
    lst_method = []
    while method < len:
        lst_method.append(lst[method])
        method += 3
    return lst_method


N1 = 0
M = 120
N2 = N1 + M

path_to_images = "C:\\Users\\Admin\\Documents\\dataset_samara\\images_test\\"
path_to_masks = "C:\\Users\\Admin\\Documents\\dataset_samara\\targets_test\\"

"""
N1 = 48000
M = 1200
N2 = N1 + M

path_to_images = "C:\\Users\\Admin\\Documents\\dataset_object\\images\\"
path_to_masks = "C:\\Users\\Admin\\Documents\\dataset_object\\targets\\"
"""

inputs = os.listdir(path_to_images)
inputs = sorted(inputs, key=lambda x: int(os.path.splitext(x)[0]))
inputs_read = [path_to_images + file for file in inputs]

masks = os.listdir(path_to_images)
masks = sorted(masks, key=lambda x: int(os.path.splitext(x)[0]))
masks_read = [path_to_masks + file for file in masks]

inputs = inputs_read[N1:N2]
targets = masks_read[N1:N2]

CRA = 0
DFN = 1
HII = 2
inputs_CRA = sort_special(CRA, inputs, M)
targets_CRA = sort_special(CRA, targets, M)
inputs_DFN = sort_special(DFN, inputs, M)
targets_DFN = sort_special(DFN, targets, M)
inputs_HII = sort_special(HII, inputs, M)
targets_HII = sort_special(HII, targets, M)

print("start validating")
model_name = 'model_unet_256_samara_extlearn_HII'
min_iou = 0.6
validate(model_name, inputs, targets, min_iou)
print(len(good)/len(inputs))
