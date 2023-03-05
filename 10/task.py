import pandas as pd
import numpy as np

# -----------------------  task 1


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ReLu(x):
    return np.maximum(0, x)


def tanh(x):
    return (2 / 1 + np.exp(-2*x)) - 1


def softMax(x):
    out = np.exp(x)
    return out / np.sum(out)


def predict(x):
    # ------ вычисление нейросети
    t1 = x @ W1 + b1
    h1 = ReLu(t1)

    t2 = h1 @ W2 + b2
    h2 = tanh(t2)

    t3 = h2 @ W3 + b3
    h3 = softMax(t3)
    return h3


# длины векторов
input_cnt = 256     # на входе
layer_cnt_1 = 256   # промежуточный 1
layer_cnt_2 = 256   # промежуточный 2
output_cnt = 4      # на выходе

# ------ создаем данные нейронок
x = np.random.rand(input_cnt)  # вход
W1 = np.random.rand(input_cnt, layer_cnt_1)  # вес для 1-го слоя
b1 = np.random.rand(layer_cnt_1)  # смещение

W2 = np.random.rand(layer_cnt_1, layer_cnt_2)  # вес для 2-го слоя
b2 = np.random.rand(layer_cnt_2)  # смещение

W3 = np.random.rand(layer_cnt_2, output_cnt)  # вес для 3-го слоя
b3 = np.random.rand(output_cnt)  # смещение


prob = predict(x)
pred_class = np.argmax(prob)
class_names = ['0', '1', '2', '3']
print("Результат: ", class_names[pred_class])

# ---------------------  task 2
# Реализовать модель с прямым проходом, состоящую из 2 свёрток с функциями активации ReLU и 2 функций MaxPool.
# Первый слой переводит из 3 каналов в 8, второй из 8 слоёв в 16. На вход подаётся изображение размера 19х19.
# (19х19x3 -> 18x18x8 -> 9x9x8 -> 8x8x16 -> 4x4x16)


class Task_2():
    def __init__(self, size_input, size_output, kernel_size, step = 1):
        self.size_input_x, self.size_input_y, self.size_input_z = size_input
        self.size_output_x, self.size_output_y, self.size_output_z = size_output
        self.kernel_size = kernel_size
        self.step = step
        self.weight = []
        for item in range(size_output[2]):
            self.weight.append(np.random.random(
                (kernel_size, kernel_size, self.size_input_z)))
        self.weight = np.array(self.weight)
        
    def reLu(self, x):
        return np.maximum(0, x)
    
    def convolution(self, x, kernel):
        k_x, k_y, _ = kernel.shape
        in_x, in_y, in_z = x.shape

        # создаем новую матрицу
        convResult = np.zeros((in_x-1, in_y-1, in_z))
    
        for channel in range(in_z):
            for v_shift in range(0, in_x - k_x + 1, self.step):
                for h_shift in range(0, in_y - k_y + 1, self.step):
                    convResult[v_shift][h_shift][channel] = np.sum(
                        x[v_shift:v_shift+k_x,h_shift:h_shift+k_y,channel] * kernel[:,:,channel]
                    )
        convResult = np.sum(convResult, axis=2)
        return (convResult)
    
    def maxPool(self, x, kernel_size, step):
        inX, inY, inC = x.shape
        maxPoolResult = np.zeros((inX//kernel_size, inY//kernel_size, inC))

        for channel in range(inC):
            for v_shift in range(0, inX - kernel_size + 1, step):
                for h_shift in range(0, inY - kernel_size + 1, step):
                    maxPoolResult[v_shift//kernel_size][h_shift//kernel_size][channel] = np.max(x[v_shift:v_shift+kernel_size,h_shift:h_shift+kernel_size,channel])
        return maxPoolResult

    def forward_conv(self, inputs):
        res = np.zeros((inputs.shape[0]-1, inputs.shape[1]-1, self.size_output_z)) 
        for index, itemWeight in enumerate(self.weight):
            applied = self.convolution(inputs, itemWeight)
            res[:, :, index] = applied[:, :]
        
        return self.reLu(res)
    
    def forward_maxPool(self, inputs):
        result = self.maxPool(inputs, self.kernel_size, self.step)
        return result


input = np.random.randint(0, 255, size=(19, 19, 3))

fuild_1 = Task_2((19,19,3), (18,18,8), 3, 1)
fuild_1 = fuild_1.forward_conv(input)
print(fuild_1.shape)

maxPool_1 = Task_2((19,19,3), (18,18,8), 2, 2)
maxPool_1 = maxPool_1.forward_maxPool(fuild_1)
print(maxPool_1.shape)

fuild_2 = Task_2((9,9,8), (8,8,16), 3, 1)
fuild_2 = fuild_2.forward_conv(maxPool_1)
print(fuild_2.shape)

maxPool_2 = Task_2((9,9,8), (8,8,16), 2, 2)
maxPool_2 = maxPool_2.forward_maxPool(fuild_2)
print(maxPool_2.shape)

# (18, 18, 8)
# (9, 9, 8)
# (8, 8, 16)
# (4, 4, 16)