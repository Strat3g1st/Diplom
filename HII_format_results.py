import cv2
import os


# отсекает в выводе HII финальное решение (четвертое изображение из склеенных)
def extract_final_images_from_HII(path):
    images = os.listdir(path)
    images = [path + file for file in images]
    for name in images:
        img = cv2.imread(name)
        h1, h2 = img.shape[0], img.shape[1]
        img = img[:, h2 - int(h2 * 0.25):h2, :]
        cv2.imwrite(name, img)


name_dir_data = "objects_add"
path1 = "C:\\Users\\maste\\Documents\\" + name_dir_data + "\\mantis\\HII\\"
path2 = "C:\\Users\\maste\\Documents\\" + name_dir_data + "\\mapple\\HII\\"
path3 = "C:\\Users\\maste\\Documents\\" + name_dir_data + "\\vinca\\HII\\"
path4 = "C:\\Users\\maste\\Documents\\" + name_dir_data + "\\wasp\\HII\\"

extract_final_images_from_HII(path1)
print('mantis formated')
extract_final_images_from_HII(path2)
print('mapple formated')
extract_final_images_from_HII(path3)
print('vinca formated')
extract_final_images_from_HII(path4)
print('wasp formated')
