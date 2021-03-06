# 1. Методы, взятые для исследования:

DFN: https://github.com/hughplay/DFNet

CRA: https://github.com/Atlas200dk/sample-imageinpainting-HiFill

HII: https://github.com/GouravWadhwa/Hypergraphs-Image-Inpainting

Заменить все test.py в этих методах на redacted версии из репозитория НИРС. Для запуска использовать параметры из DFN_param.txt, HII_param.txt, CRA_param.txt. Причем CRA_param.txt - это переменные в самом исходном коде test.py. Для HII использованные параметры в виде conf_<name>.txt - прилагаются в репозитории. Сгенерировать заново с заданными путями их можно с помощью conf_gen.py В случае CRA использовался код в папке GPU_CPU. Также как и в НИРС надо заменить версии кода методов на отредактированные в репозитории.

Запуск на GPU всех методов проходит успешно.

Pretrained models

Для DFNet взять places2: https://drive.google.com/drive/folders/1lKJg__prvJTOdgmg9ZDF9II8B1C3YSkN

Для CRA взять: https://github.com/Atlas200dk/sample-imageinpainting-HiFill/tree/master/GPU_CPU/pb

Для HII взять все отсюда: https://drive.google.com/drive/folders/1dk1zSm1FxZVaafOtvoud8aAdZ6Ubs4oU 

# 2. Подготовка датасета

Изначально формируется вручную или с помощью скрипта система папок, которая содержится в папке с произвольным названием. В коде это переменная name_dir_data. В этой папке может содержаться ряд папок, каждая из которых обозначает определенный вырезанный объект на изображении. В данном случае это 4 объекта: mantis (богомол), mapple (кленвый лист), vinca (цветок барвинка), wasp (оса). 
  
Изображения и их маски взяты отсюда: https://www.kaggle.com/metavision/datasets

В каждой папке объекта есть папки images, masks, CRA, DFN, HII. images - это изображения с объектами; masks - сегментированные маски объектов; CRA, DFN, HII - изображения с вырезанными и ретушированными разными методами объектами. 
  
Сначала необходимо скопировать изображения и маски объектов в соответствующие папки. Далее применисть скрипт prepare_data_object.py. Затем запустить DFN и CRA на сгенерированных изображениях. Затем применить скрипт invert.py, который меняет местами черную и белую область в масках. После этого запустить HII. Далее применить скрипт HII_format_results.py.

Затем необходимо запустить скрипт augmentation.py, который объединяет полученные изображения и маски в единый датасет с двумя папками - images и masks, а также применяет повороты под разными углами с целью добиться пространственной инвариантности при обучении нейросети.

# 3. Обучение и тестирование нейросетей
  
Для обучения нейросети на датасете используется скрипт train_model.py. В нем импортируется один из классов CNNNet, означающий определенную архитектуру сверточной нейросети. Импорт может происходить из decoder_32.py, decoder_64.py, decoder_128.py, decoder_256.py, UNet256.py. Также применялась попытка обучить архитектуры SegNet256.py и SegNet128.py, но функция потерь в них быстро застывает в определенном значении, возможно из-за умирания ReLU, причем использование LeakyReLU и Dropout не помогает. Видимо слишком глубокие сети тоже снижают эффективность обучения. После обучения нейросеть сохраняется в указанный файл.

Для проверки нейросети на тестовых данных используется скрипт validate_model.py. Выводятся значения функции потерь, а затем итоговый процент случаев, в котором функция потерь была ниже заданного порога.
  
Для генерации масок с детектированными случаями ретуширования с помощью предварительно обученной нейросети используется скрипт load_model.py для определенной папки с изображениями.
