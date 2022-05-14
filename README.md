# 1. Методы, взятые для исследования:

DFN: https://github.com/hughplay/DFNet

CRA: https://github.com/Atlas200dk/sample-imageinpainting-HiFill

HII: https://github.com/GouravWadhwa/Hypergraphs-Image-Inpainting

Заменить все test.py в этих методах на redacted версии из репозитория НИРС. Для запуска использовать параметры из DFN_param.txt, HII_param.txt, CRA_param.txt. Причем CRA_param.txt - это переменные в самом исходном коде test.py. Для HII использованные параметры в виде conf_<name>.txt - прилагаются в репозитории. Сгенерировать заново с заданными путями их можно с помощью conf_gen.py В случае CRA использовался код в папке GPU_CPU.

Запуск на GPU всех методов проходит успешно.

Pretrained models

Для DFNet взять places2: https://drive.google.com/drive/folders/1lKJg__prvJTOdgmg9ZDF9II8B1C3YSkN

Для CRA взять: https://github.com/Atlas200dk/sample-imageinpainting-HiFill/tree/master/GPU_CPU/pb

Для HII взять все отсюда: https://drive.google.com/drive/folders/1dk1zSm1FxZVaafOtvoud8aAdZ6Ubs4oU 

# 2. Подготовка датасета

Изначально формируется вручную или с помощью скрипта система папок, которая содержится в папке с произвольным названием. В коде это переменная name_dir_data. В этой папке может содержаться ряд папок, каждая из которых обозначает определенный вырезанный объект на изображении. В данном случае это 4 объекта: mantis (богомол), mapple (кленвый лист), vinca (цветок барвинка), wasp (оса). 
  
Изображения и их маски взяты отсюда: https://www.kaggle.com/metavision/datasets

В каждой папке объекта есть папки images, masks, CRA, DFN, HII. images - это изображения с объектами; masks - сегментированные маски объектов; CRA, DFN, HII - изображения с вырезанными и ретушированными разными методами объектами.
