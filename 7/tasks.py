import cv2
import os
import numpy as np
from dataclasses import dataclass
from os.path import join as pjoin
import matplotlib.pyplot as plt
import random as rand
import numpy.random as nrand

image = cv2.imread('./data/photo.jpg', 0)

# ----- 1 Для датасета Nails segmentation создать генератор, который на каждой итерации возвращает пару списков из заданного количества 
# (аргумент функции) изображений и масок к ним (итератор должен перемешивать примеры).
# ----- 2 Добавить в генератор случайную аугментацию (каждая применяется случайно). После преобразований все изображения должны иметь одинаковый размер. 
# Обратите внимание, что большинство преобразований должны применяться одинаково к изображению и маске
# - Поворот на случайный угол
# - Отражение по вертикали, горизонтали
# - Вырезание части изображения
# - Размытие

@dataclass
class Pairs:
    name : str
    img   : cv2.Mat
    mask : cv2.Mat

    def __repr__(self):
        return self.name


def loadImgsWMasks(imagesFolder : str, masksFolder : str) -> list[Pairs]:
    nails = []
    for name in os.listdir(imagesFolder):
        if (os.path.isfile(name)):
            continue
        image = cv2.imread(pjoin(imagesFolder, name))
        mask  = cv2.imread(pjoin(masksFolder, name))
        nails.append(Pairs(name, image, mask))
    return nails

# Изменить размер картинки
def resize(pair: Pairs, wight:int, hight:int) -> Pairs:
    return Pairs(pair.name, cv2.resize(pair.img, dsize=(wight, hight)), cv2.resize(pair.mask, dsize=(wight, hight)))

# - Поворот на случайный угол
def rotateImg(pair:Pairs) -> Pairs:
    angle = rand.randint(0, 360)
    height, width, _  = pair.img.shape
    rot_mat = cv2.getRotationMatrix2D((height/2, width/2), angle, 1.0)
    return Pairs(pair.name, 
                    cv2.warpAffine(pair.img, rot_mat, (height, width), flags=cv2.INTER_LINEAR), 
                    cv2.warpAffine(pair.mask, rot_mat, (height, width), flags=cv2.INTER_LINEAR))
    
# - Отражение по вертикали, горизонтали
def mirrorImg(pair:Pairs) -> Pairs:
    isVertical = rand.randint(0,1)
    if isVertical == 1:
        return Pairs(pair.name, pair.img[::-1], pair.mask[::-1])
    else:
        return Pairs(pair.name, pair.img[:,::-1], pair.mask[:,::-1])

# - Вырезание части изображения
def cutImg(pair:Pairs) -> Pairs:
    np.random.seed(43)
    bbox = nrand.randint(0, pair.img.shape[1]/2, size=4)
    return Pairs(pair.name,
                    pair.img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]],
                    pair.mask[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]])

# - Размытие
def bluerImg(pair:Pairs) -> Pairs:
    blur = rand.randint(3, 30)
    return Pairs(pair.name, cv2.blur(pair.img, (blur, blur)), cv2.blur(pair.mask, (blur, blur)))

# получить набор данных
def getDataSet(pairs:list[Pairs], cnt) -> list[Pairs]:
    listNails = []
    for itemPair in range(cnt):
        listNails.append(pairs[itemPair])
    return listNails


# --------- MAIN ----------
imagesFolder = "assets/nails/images"
masksFolder  = "assets/nails/labels"
action=[rotateImg, mirrorImg, cutImg, bluerImg]
nails = loadImgsWMasks(imagesFolder, masksFolder)

for itemPair in getDataSet(nails, 5):
    indexAction = rand.randint(0, len(action)-1)
    newPair = action[indexAction](resize(itemPair, 512, 512))
    print(indexAction)

    cv2.imshow("Nail", newPair.img)
    cv2.imshow("Mask", newPair.mask)
    key = cv2.waitKey(0) & 0xff
    if key == ord('Q') or key == ord('q') or key == 27:
        break

cv2.destroyAllWindows()