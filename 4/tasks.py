import numpy as np

# -------- 1 TASK ( Отсортировать значения массива по частоте встречания. )
arr = np.random.randint(low=0, high=10, size=10)
print(*arr)
# получить массив уникальных значение, а так же счетчик
arr, counts = np.unique(arr, return_counts=True)
# print(arr, counts)
arr = sorted(zip(counts, arr), reverse=True)
print(*[x for _,x in arr])

# -------- 2 TASK ( Дана картинка высоты h, ширины w, тип данных np.uint8 (от 0 до 255). Найти количество уникальных цветов. )
w=2
h=2
# генерируем (3 так как rgb)
img = np.random.randint(low=0, high=256, dtype=np.uint8, size=(h,w,3))
print(*img)
amount = np.unique(img, axis=0)
# print(amount)
print(len(amount))

# -------- 3 TASK ( Написать функцию, вычислающую плавающее среднее вектора )
arr = np.random.randint(low=0, high=10, size=10)
print(arr)
res = []
for i in range(0, len(arr)-3):
    res.append(arr[i:i+3].sum()/3)
print(res)

# -------- 4 TASK ( Дана матрица (n, 3). Вывести те тройки чисел, которые являются длинами сторон треугольника. )
# проверка условия
def is_tri(arr):
    return sum(arr) - 2 * max(arr) > 0

n = 5
arr = np.random.randint(low=0, high=100, size=(n, 3))
print(arr)
triangs = zip(arr, np.apply_along_axis(is_tri, 1, arr))
for P, isTri in triangs: 
    if isTri:
        print(str(P) + "  " + str(isTri))


