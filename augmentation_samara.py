import os
import shutil
from PIL import Image


def rot0(path_read, path_save):
    shutil.copy(path_read, path_save)
    return


def rot90(path_read, path_save):
    img = Image.open(path_read)
    img = img.rotate(90)
    img.save(path_save)
    return


def rot180(path_read, path_save):
    img = Image.open(path_read)
    img = img.rotate(180)
    img.save(path_save)
    return


def rot270(path_read, path_save):
    img = Image.open(path_read)
    img = img.rotate(270)
    img.save(path_save)
    return


def augmentation(masks, method1, method2, method3, path_save_images, path_save_masks, funcs):
    M = []
    M1 = []
    M2 = []
    M3 = []
    for i in range(len(masks)):
        M.append(os.listdir(masks[i]))
        M1.append(os.listdir(method1[i]))
        M2.append(os.listdir(method2[i]))
        M3.append(os.listdir(method3[i]))
    id = 1
    print(len(M[0]))
    for i in range(len(M[0])):
        print(i)
        for j in range(len(M)):
            for f in funcs:
                f(masks[j] + M[j][i], path_save_masks + str(id) + '.png')
                f(method1[j] + M1[j][i], path_save_images + str(id) + '.png')
                id += 1
                f(masks[j] + M[j][i], path_save_masks + str(id) + '.png')
                f(method2[j] + M2[j][i], path_save_images + str(id) + '.png')
                id += 1
                f(masks[j] + M[j][i], path_save_masks + str(id) + '.png')
                f(method3[j] + M3[j][i], path_save_images + str(id) + '.png')
                id += 1
    return


new_dataset = "dataset_samara_all"
path_save_images = "C:\\Users\\maste\\Documents\\" + new_dataset + "\\images\\"
path_save_masks = "C:\\Users\\maste\\Documents\\" + new_dataset + "\\targets\\"
# os.mkdir("C:\\Users\\maste\\Documents\\" + new_dataset)
# os.mkdir(path_save_images)
# os.mkdir(path_save_masks)

name_dir_data = "dataset_samara"
path_mask_1 = "C:\\Users\\maste\\Documents\\" + name_dir_data + "\\masks\\"
MASKS = [path_mask_1]

path_cra_1 = "C:\\Users\\maste\\Documents\\" + name_dir_data + "\\CRA\\"
CRA = [path_cra_1]

path_dfn_1 = "C:\\Users\\maste\\Documents\\" + name_dir_data + "\\DFN\\result\\"
DFN = [path_dfn_1]

path_hii_1 = "C:\\Users\\maste\\Documents\\" + name_dir_data + "\\HII\\"
HII = [path_hii_1]

augmentation(MASKS, CRA, DFN, HII, path_save_images, path_save_masks, [rot0, rot90, rot180, rot270])
