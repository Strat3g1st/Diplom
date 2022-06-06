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


path_save_images = "C:\\Users\\maste\\Documents\\dataset\\images\\"
path_save_masks = "C:\\Users\\maste\\Documents\\dataset\\targets\\"

path_mask_1 = "C:\\Users\\maste\\Documents\\materials\\masks\\places2\\mask_frame\\0.1_\\"
path_mask_2 = "C:\\Users\\maste\\Documents\\materials\\masks\\places2\\mask_grid\\0.1_\\"
path_mask_3 = "C:\\Users\\maste\\Documents\\materials\\masks\\places2\\mask_half\\0.1_\\"
path_mask_4 = "C:\\Users\\maste\\Documents\\materials\\masks\\places2\\mask_half\\0.2_\\"
path_mask_5 = "C:\\Users\\maste\\Documents\\materials\\masks\\places2\\mask_rectangle\\0.1_\\"
path_mask_6 = "C:\\Users\\maste\\Documents\\materials\\masks\\places2\\mask_rectangle\\0.2_\\"
MASKS = [path_mask_1, path_mask_2, path_mask_3, path_mask_4, path_mask_5, path_mask_6]

path_cra_1 = "C:\\Users\\maste\\Documents\\materials\\results\\places2\\CRA\\recovery_frame\\0.1_\\"
path_cra_2 = "C:\\Users\\maste\\Documents\\materials\\results\\places2\\CRA\\recovery_grid\\0.1_\\"
path_cra_3 = "C:\\Users\\maste\\Documents\\materials\\results\\places2\\CRA\\recovery_half\\0.1_\\"
path_cra_4 = "C:\\Users\\maste\\Documents\\materials\\results\\places2\\CRA\\recovery_half\\0.2_\\"
path_cra_5 = "C:\\Users\\maste\\Documents\\materials\\results\\places2\\CRA\\recovery_rectangle\\0.1_\\"
path_cra_6 = "C:\\Users\\maste\\Documents\\materials\\results\\places2\\CRA\\recovery_rectangle\\0.2_\\"
CRA = [path_cra_1, path_cra_2, path_cra_3, path_cra_4, path_cra_5, path_cra_6]

path_dfn_1 = "C:\\Users\\maste\\Documents\\materials\\results\\places2\\DFN\\recovery_frame\\0.1_\\result\\"
path_dfn_2 = "C:\\Users\\maste\\Documents\\materials\\results\\places2\\DFN\\recovery_grid\\0.1_\\result\\"
path_dfn_3 = "C:\\Users\\maste\\Documents\\materials\\results\\places2\\DFN\\recovery_half\\0.1_\\result\\"
path_dfn_4 = "C:\\Users\\maste\\Documents\\materials\\results\\places2\\DFN\\recovery_half\\0.2_\\result\\"
path_dfn_5 = "C:\\Users\\maste\\Documents\\materials\\results\\places2\\DFN\\recovery_rectangle\\0.1_\\result\\"
path_dfn_6 = "C:\\Users\\maste\\Documents\\materials\\results\\places2\\DFN\\recovery_rectangle\\0.2_\\result\\"
DFN = [path_dfn_1, path_dfn_2, path_dfn_3, path_dfn_4, path_dfn_5, path_dfn_6]

path_hii_1 = "C:\\Users\\maste\\Documents\\materials\\results\\places2\\HII\\recovery_frame\\0.1_\\"
path_hii_2 = "C:\\Users\\maste\\Documents\\materials\\results\\places2\\HII\\recovery_grid\\0.1_\\"
path_hii_3 = "C:\\Users\\maste\\Documents\\materials\\results\\places2\\HII\\recovery_half\\0.1_\\"
path_hii_4 = "C:\\Users\\maste\\Documents\\materials\\results\\places2\\HII\\recovery_half\\0.2_\\"
path_hii_5 = "C:\\Users\\maste\\Documents\\materials\\results\\places2\\HII\\recovery_rectangle\\0.1_\\"
path_hii_6 = "C:\\Users\\maste\\Documents\\materials\\results\\places2\\HII\\recovery_rectangle\\0.2_\\"
HII = [path_hii_1, path_hii_2, path_hii_3, path_hii_4, path_hii_5, path_hii_6]

augmentation(MASKS, CRA, DFN, HII, path_save_images, path_save_masks, [rot0, rot90, rot180, rot270])
