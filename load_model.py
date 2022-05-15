import os
import torch
import numpy as np
from PIL import Image
from UNet256 import CNNNet


def invert(img):
    img[img < 127] = 1
    img[img >= 127] = 2
    img[img == 1] = 255
    img[img == 2] = 0
    return img


def detect(model, path_read, path_write):
    test_img = Image.open(path_read)
    test_img = test_img.convert('RGB')
    test_img = np.asarray(test_img)
    test_img = np.expand_dims(test_img, axis=0)
    test_img = torch.from_numpy(test_img)
    test_img = test_img.type(torch.FloatTensor)
    test_img = test_img.reshape(test_img.shape[0], test_img.shape[3], test_img.shape[1], test_img.shape[2])

    result_img = model.forward(test_img)
    result_img = torch.max(result_img, 1)[0]
    result_img = result_img.detach().numpy()
    result_img = np.squeeze(result_img, axis=0)
    result_img = result_img.astype(np.uint8)
    result_img = invert(result_img)
    result = Image.fromarray(result_img)
    result.save(path_write)


model = CNNNet()
model.load_state_dict(torch.load("model_unet_256_x5"))
model.eval()

N = 72000
path_to_images = "C:\\Users\\Admin\\Documents\\dataset_object\\images\\"
path_to_save = "C:\\Users\\Admin\\Documents\\dataset_object\\predicted\\"
inputs = os.listdir(path_to_images)
inputs = sorted(inputs, key=lambda x: int(os.path.splitext(x)[0]))
inputs_read = [path_to_images + file for file in inputs]

os.mkdir(path_to_save)

for i in range(len(inputs)):
    name_read = inputs_read[i]
    name_write = path_to_save + inputs[i]
    detect(model, name_read, name_write)
