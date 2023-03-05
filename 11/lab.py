import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# 11 тема - Задание: "Написать SimpleModel другим способом. Использовать model = nn.Sequential() https://pytorch.org/tutorials/beginner/nn_tutorial.html?highlight=mnist ". 
# + hw.pdf ( https://github.com/d-pack/LessonsPAK/blob/main/s14/hw.pdf )


# Простая модель
class SimpleModel(nn.Module):
    def __init__(self, in_ch, out_ch):
        """Регистрация блоков"""
        super().__init__()
        self.fc1 = nn.Linear(in_ch, 32)  # Полносвязный слой 1
        self.fc2 = nn.Linear(32, out_ch, bias=False)  # Полносвязный слой 2
        self.relu = nn.ReLU()  # Функция активации
        
    def forward(self, x):
        """Прямой проход"""
        h = self.fc1(x)
        h = self.relu(h)
        h = self.fc2(h)
        y = self.relu(h)
        return y

model = SimpleModel(64, 10)

print(list(model.parameters()))