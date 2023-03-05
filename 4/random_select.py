import argparse
import numpy as np

# Есть два набора данных: реальные и синтетические. Допустим, мы хотим обучить некоторую ML модель на смеси реальных и синтетических данных. 
# При этом синтетические данные должны браться с вероятностью P. Важно сохранять порядок входных чисел. 
# Например: Для массивов: [1,2,3,4,5,7,8,9,10] и [-1,-2,-3,-4,-5,-6,-7,-8,-9,-10] и P=0.2
# Один из вариантов возвращаемого значения: [1,-2,3,4,-5,6,7,8,9,10]

# Массивы реальных и синтетических данных одинаковой длины.

# Реализовать скрипт random_select.py
# Входные параметры скрипта: пути к двум файлам со списком целых чисел в каждом. Например file_1.txt содержит:
# 1 2 3 4 5 6 7
# а file_2.txt
# -1 -2 -3 -4 -5 -6 -7

def first_var(arr1, arr2, p:float):
    Arr12 = np.stack((arr1,arr2), axis=1)
    #print(Arr12)
    res = map(lambda x: np.random.choice([int(x[0]), int(x[1])], size=1, p=[1-p, p])[0], Arr12)
    print(list(res))
    
def second_var(arr1, arr2, p:float):
    arr = np.random.random(size=len(arr1))
    print(np.where([arr>=p], arr1,arr2).ravel())

def tri_var(arr1, arr2, p:float):
    Arr12 = zip(list(arr1), list(arr2))
    lab = lambda x: x[0] if np.random.random()>=p else x[1]
    res = map(lab, Arr12)
    print(list(res))


# main
parser = argparse.ArgumentParser()
parser.add_argument("fileName1", metavar="file name #1", nargs='?', help="Name file #1")
parser.add_argument("fileName2", metavar="file name #2", nargs='?', help="Name file #2")
parser.add_argument("P", metavar="predictably", type=float, nargs='?', help="predictably [0,1]")
args = parser.parse_args()

f1 = args.fileName1
f2 = args.fileName2
p = args.P
start = True
content1, content2 = [], []

# проверка вероятности
if (p > 1 or p < 0):
    ValueError("ERROR! P shut be range from 0 to 1")
    start=False
else:
    with open(f1, 'r') as f:
        content1 =  np.array(f.readline().split())
    with open(f2, 'r') as f:
        content2 =  np.array(f.readline().split())
    
    if (len(content1) != len(content2)):
        ValueError("ERROR! len arg file1 != file 2")
    else:
        first_var(content1, content2, p)
        second_var(content1, content2, p)
        tri_var(content1, content2, p)




