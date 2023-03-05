import pandas as pd
import numpy as np

# -------  функция потерь
# - функция, оценивающая качество работы всей модели

# ------- скорость обучения learningRate,
# Скорость обучения — это просто самый важный гиперпараметр в нейронной сети что бы минимизировать ошибку
# В начале процесса обучения параметры сети (веса и смещения) инициализируются случайными значениями, но это не оптимальные значения,
# дающие минимальную ошибку или потери. Поэтому мы и дальше продолжаем тренировочный процесс.

# На каждом этапе обучения (итерации) сети результаты, рассчитанные на этапе прямого распространения, сравниваются с истинными (факическими)
# значениями для расчета оценки ошибки или потери.
# Затем эта ошибка распространяется обратно на этапе обратного распространения, чтобы скорректировать начальные значения весов и смещений,
# чтобы минимизировать ошибку на следующих этапах обучения. Продолжаем обучение, пока не получим минимальную ошибку.

# Чтобы вычислить ошибку между прогнозируемыми значениями и фактическими значениями на каждом шаге обучения, требуется функция потерь .
# Во время обучения наша цель — максимально минимизировать функцию потерь. Для минимизации функции потерь нам нужен оптимизатор (алгоритм оптимизации).


class MyNeuron:
    def __init__(self, size_input, size_output, learning_rate=0.1):
        self.size_input = size_input
        self.size_output = size_output
        self.learning_rate = learning_rate
        # Инициализируйте веса и смещения случайным образом.
        self.wight = np.random.uniform(size=(size_input, size_output))
        self.bias = np.random.uniform(size=(1, size_output))

    # функция активации
    # определяет активность нейрона
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # функция ошибки
    # называется разница между правильными ответами и ответами, которая выдает нейронная сеть на обучающем и контрольном датасете.
    def sigmoidDerivative(self, x):
        return x * (1 - x)

    def loss(self, error):
        return error * self.sigmoidDerivative(self.last_value)

    # прямой проход по нейронной сети,
    # должен реализовать логику работу нейрона,
    # - умножение входа на вес, сложение и функцию активации сигмоиду.
    def forwaed(self, x):
        self.last_value = self.sigmoid((x @ self.wight) + self.bias)
        return self.last_value

    # обратный проход по нейронной сети,
    # должен реализовать взятие производной от сишмоиды и используя состояние нейрона
    # обновить его веса
    def backward(self, x: np.ndarray, loss_value):
        self.wight += x.T @ loss_value * self.learning_rate
        self.bias += np.sum(loss_value, axis=0, keepdims=True) * self.learning_rate


class MyModel():
    def __init__(self, size_input, hidden_size, size_output):
        self.size_input = size_input
        self.hidden_size = hidden_size
        self.size_output = size_output
        # создаем сетку нейронов где размер это количесвто стоков, истоков, скорость обучения lr
        self.hidden_layer = MyNeuron(size_input, hidden_size)
        self.output_layer = MyNeuron(hidden_size, size_output)

    def forwaed(self, x):
        self.hidden_val = self.hidden_layer.forwaed(x)
        self.last_val = self.output_layer.forwaed(self.hidden_val)
        return self.last_val

    def backward(self, x, error):
        loss_layer = self.output_layer.loss(error)
        error_hidden_layer = loss_layer @ self.output_layer.wight.T
        loss_hidden_layer = self.hidden_layer.loss(error_hidden_layer)

        self.hidden_layer.backward(x, loss_hidden_layer)
        self.output_layer.backward(self.hidden_val, loss_layer)

    def get_result(self):
        return self.last_val


# ----- MAIN -----
input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
res_output = np.array([[0], [1], [1], [0]])
epochs = 10000

# по статье у нас 2 слоя входа, 2 скрытий и 1 на выход
model = MyModel(2, 2, 1)

# цикл обучения данных
for item in range(epochs):
    res = model.forwaed(input)
    error = res_output - res
    model.backward(input, error)

print(model.get_result())

# [[0.05747914]    0
#  [0.94694962]    1
#  [0.94690278]    1
#  [0.05732657]]   0
