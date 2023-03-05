import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# 1 Задание: "Написать SimpleModel другим способом. 
# Использовать model = nn.Sequential() https://pytorch.org/tutorials/beginner/nn_tutorial.html?highlight=mnist ". 
# 2 hw.pdf ( https://github.com/d-pack/LessonsPAK/blob/main/s14/hw.pdf )

# 1 "Написать SimpleModel другим способом. Использовать model = nn.Sequential()
in_ch = 64
out_ch = 10

model = nn.Sequential(
    nn.Linear(in_ch, 32),
    nn.ReLU(),
    nn.Linear(32, out_ch, bias=False),
    nn.ReLU(),
)

# print(list(model.parameters()))

# 2 hw.pdf ( https://github.com/d-pack/LessonsPAK/blob/main/s14/hw.pdf )

# 2.1
task_1 = torch.tensor([[[1,-1,0],[2,0,1],[1,1,0]], [[2,0,1],[4,0,2],[2,1,1]], [[1,1,0],[2,0,1],[1,-1,0]]], dtype=torch.float32)
conv_1 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=2, stride=1, padding=1, dilation=2)
print(conv_1(task_1))
# tensor([[[0.1836, 0.6339, 0.1836],
#          [0.1247, 0.3967, 0.3008],
#          [0.1836, 0.3879, 0.1836]]], grad_fn=<SqueezeBackward1>)

task_1_2 = torch.tensor([[[1,0,1],[0,-1,-1]], [[0,-1,0],[1,0,0]]], dtype=torch.float32).reshape(3,2,2)
conv_1_2 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=2, stride=1, padding=1, dilation=2)
print(conv_1_2(task_1_2))
# tensor([[[-0.3917, -0.0654],
#          [-0.2164, -0.3930]]], grad_fn=<SqueezeBackward1>)

# 2.2
task_2 = torch.empty((1,1,1))
conv_2 = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1),
    nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1, dilation=1),
    nn.MaxPool2d(2, padding=0, stride=2),
    nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=100500, dilation=2),
    nn.MaxPool2d(3, padding=28, stride=3),
)

# поле = stride2 x поле2 + (kernel_size2 - stride2)
# p1 = 2 * 
# print(conv_2(task_2).shape)

# 2.3
task_3 = torch.empty((1,224,224))
conv_3 = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=1, kernel_size=7, stride=2, padding=3),
    nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1),
    nn.MaxPool2d(2, padding=0, stride=2),
    nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1, dilation=3),
)
print(conv_3(task_3).shape)
# torch.Size([1, 26, 26])