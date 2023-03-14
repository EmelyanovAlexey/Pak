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
task_1 = torch.tensor([[[1, -1, 0], [2, 0, 1], [1, 1, 0]], [[2, 0, 1], [4, 0, 2],
                      [2, 1, 1]], [[1, 1, 0], [2, 0, 1], [1, -1, 0]]], dtype=torch.float32)
kernel = torch.Tensor(
    [[[1, 0],
      [0, 1]],
     [[0, -1],
      [-1, 0]],
     [[1, -1],
        [0, 0]]
     ]).view(1, 3, 2, 2)
res_task_1 = nn.functional.conv2d(task_1, kernel, padding=1, dilation=2)
# print(res_task_1)
# tensor([[[ 0., -3.,  0.],
#          [ 0., -1., -1.],
#          [ 0.,  1.,  0.]]])

# 2.2
# task_2 = torch.empty((1, 1, 1))
# conv_2 = nn.Sequential(
#     nn.Conv2d(in_channels=1, out_channels=1, kernel_size=7,
#               stride=1, padding=3, dilation=1),
#     nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3,
#               stride=2, padding=1, dilation=1),
#     nn.MaxPool2d(2, padding=0, stride=2),
#     nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3,
#               stride=1, padding=100500, dilation=2),
#     nn.MaxPool2d(3, padding=28, stride=3),
# )


class Layers:
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, dilation):
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.stride = stride

    def res_field(lays):
        stris = 1
        res = 1
        for i in lays:
            stris *= i.stride
            res += i.dilation * (i.kernel_size - 1) * stris
        return res

layers = [Layers(1, 1, 7, 3, 1, 1), Layers(1, 1, 3, 1, 2, 1), Layers(
    1, 1, 2, 0, 2, 1), Layers(1, 1, 3, 100500, 1, 2), Layers(1, 1, 3, 28, 3, 1)]

Layers.res_field(layers)
# torch.Size([1, 26, 26])

# поле = stride2 x поле2 + (kernel_size2 - stride2)
# p1 = 2 *
# print(conv_2(task_2).shape)

# 2.3
task_3 = torch.empty((1, 224, 224))
conv_3 = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=1,
              kernel_size=7, stride=2, padding=3),
    nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1),
    nn.MaxPool2d(2, padding=0, stride=2),
    nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3,
              stride=2, padding=1, dilation=3),
)
print(conv_3(task_3).shape)
# torch.Size([1, 26, 26])
