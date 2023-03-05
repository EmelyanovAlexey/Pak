# Лабораторная 6.1
# Программа должна реализовывать следующий функционал:

# Покадровое получение видеопотока с камеры. Использовать камеру ноутбука, вебкамеру или записать видео файл с вебкамеры товарища и использовать его.
# Реализовать обнаружение движения в видеопотоке: попарно сравнивать текущий и предыдущий кадры. 
# (Если вы сможете синтезировать более сложный алгоритм, устойчивый к шумам вебкамеры - будет совсем хорошо)
# По мере проигрывания видео в отдельном окне отрисовывать двухцветную карту с результатом: красное - есть движение, зелёное - нет движения

# Лабораторная 6.2
# Накладывать карту движения на исходный кадр и отрисовывать в отдельном окне результат. Должно получиться примерно как на скриншоте выше.
# Добавить таймер, по которому включается и выключается обнаружение движения. 
# О текущем режиме программы сообщать текстом с краю изображения: “Красный свет” - движение обнаруживается, “Зелёный свет” - движение не обнаруживается.

import numpy as np
import cv2
# с таймером помогли
class MovementTimer:
    timer = 0
    def __init__(self, cyclePeriod = 5):
        self.cyclePeriod = cyclePeriod
    
    def timeIT(self):
        self.timer += 1
        if self.timer == self.cyclePeriod:
            self.timer = 0
            return True
        return False

class FrameRenderer():
    def __init__(self, initialframe = None , cyclePeriod = 10):
        self.timer = MovementTimer(cyclePeriod)
        self.prevFrame = initialframe
        self.state = False

    def cycle(self, nextFrame  : cv2.Mat):
        if self.timer.timeIT():
            self.state = not self.state
        if self.state:
            # поиск движущихся объектов
            # вычитание
            search_move = cv2.absdiff(nextFrame, self.prevFrame)
            # убираем шумы для четкого определения
            search_move = cv2.GaussianBlur(cv2.cvtColor(search_move, cv2.COLOR_BGR2GRAY), (5,5), 0)
            # рисуем границы
            search_move  = cv2.dilate(search_move, None, iterations=2)
            # выделение
            search_move = cv2.threshold(search_move, thresh=20, maxval=255, type=cv2.THRESH_BINARY)[1]
            contours, _ = cv2.findContours(search_move, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            out_frame   = self.prevFrame.copy()
            movement_marks = np.empty_like(out_frame, np.uint8)
            for con in contours:
                (x,y,w,h) = cv2.boundingRect(con)
                # рисуем прямоугольник
                if cv2.contourArea(con) > 120:
                    cv2.rectangle(movement_marks, (x,y), (x+w, y+h), (0,0,255), cv2.FILLED)
            mask = movement_marks.astype(bool)
            out_frame[mask] = cv2.addWeighted(out_frame, 0.4, movement_marks, 0.6, 0)[mask]
            frameMOne = out_frame
        else:
            frameMOne = nextFrame
        
        self.prevFrame = nextFrame.copy()
        cv2.imshow('video', nextFrame)
        if self.state:
            cv2.imshow('trackVideo', cv2.putText(frameMOne, "findMoveObj", (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1))
        else:
            cv2.imshow('trackVideo', frameMOne)

# чтение
cap = cv2.VideoCapture("./video/FroggerHighway.mp4")
ret, prev = cap.read()
renderer = FrameRenderer(initialframe=prev, cyclePeriod=60)

while True:
    key = cv2.waitKey(20) & 0xff
    ret, frame = cap.read()
    if key == 27 or not ret: break
    renderer.cycle(frame)
    
cv2.destroyAllWindows()
cap.release()