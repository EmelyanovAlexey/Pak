# import numpy as np
# import cv2

# ---------------
# фото
# img = cv2.imread('./anim.jpg')
# cv2.imshow('anim', img)
# cv2.waitKey(0)

# видео
# cap = cv2.VideoCapture(1)
# while True:
#     success, img = cap.read()
#     cv2.imshow('cap', img)
#     if (cv2.waitKey(1) & 0xFF == ord('q')):
#         break

# ---------------

# img = cv2.imread('./img/anim.jpg')
# # перевод в серый
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # вычисление порога 
# img = cv2.Canny(img, 150, 150)
# # ширина обводки
# kernel = np.ones((5, 5), np.uint8)
# img = cv2.dilate(img, kernel, iterations=1)
# img = cv2.erode(img, kernel, iterations=1)

# cv2.imshow('anim', img)
# cv2.waitKey(0)

# 