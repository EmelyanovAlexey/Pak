import numpy as np
import cv2
import os
from os.path import join as pjoin

imagesFolder = "./img"
masksFolder  = "./labels"
indexPicture = 0
listImages = []

for name in os.listdir(imagesFolder):
    if (os.path.isfile(name)):
        continue
    image = cv2.imread(pjoin(imagesFolder, name))
    mask  = cv2.imread(pjoin(masksFolder, name))
    maskProc = cv2.cvtColor(cv2.Canny(mask, 10, 200), cv2.COLOR_GRAY2BGR)
    maskProc[np.where((maskProc!=[0, 0, 0]).all(axis=2))] = [0, 255, 0]
    maskProc = cv2.dilate(maskProc, np.ones((3,3)), iterations=1)
    listImages.append(cv2.addWeighted(image, 0.7, maskProc, 0.3, 0))
    
while True:
    # text
    imagetoShow = cv2.putText(listImages[indexPicture],'d <- a -> e exit', (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imshow("Picture", imagetoShow)
    key = cv2.waitKey(0) & 0xff
    if key == ord('a'):
        indexPicture = (indexPicture - 1) % len(listImages)
    if key == ord('d'):
        indexPicture = (indexPicture + 1) % len(listImages)
    if key == ord('e') or key == 27:
        break

cv2.destroyAllWindows()

# работа с видео
cap = cv2.VideoCapture("video/FroggerHighway`.mp4")
while True:
    success, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('cap', img)
    if (cv2.waitKey(20) & 0xFF == ord('e')):
        break

cv2.destroyAllWindows()