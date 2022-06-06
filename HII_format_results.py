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


name_dir_data = "dataset_samara\\"
path = "C:\\Users\\maste\\Documents\\" + name_dir_data + "HII\\"

extract_final_images_from_HII(path)
print('mantis formated')
