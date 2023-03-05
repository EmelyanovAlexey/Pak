import math
import random
import os

# Сгенерировать случайное трехзначное число. Вычислить сумму его цифр
def program1(l, r):
    sum = 0
    number = random.randint(l, r)
    print("number = " + str(number))
    for i in str(number):
        sum += int(i)
    print("summa = " + str(sum))

# Задаётся радиус сферы, найти площадь её поверхности и объём.
def program3(r):
    v = (4 * math.pi * math.pow(r, 3))/3
    s = 4 * math.pi * math.pow(r, 2)
    print("V = " + str(v))
    print("S = " + str(s))

# Задаётся год. Определить, является ли он високосным
def program4(year: int):
    if year % 4 == 0 and year % 100 != 0 or year % 400 == 0:
        print("Весокосный")
    else:
        print("Не весокосный")

def program5(l, r):
    for i in range(l,r):
        l = 1
        for j in range(2,i): 
            if i%j==0:
                l = 0
                break
        if l:
            print(i)

def program6(x, y):
    sum = x
    for i in range(0,y):
        sum += (sum*10/100)
    print("sum" + str(sum))


def program7():
    print("eefe")
    path ="C:\Intel"
    filelist = []
    for root, dirs, files in os.walk(path):
        for file in files:
            filelist.append(os.path.join(root,file))
    for name in filelist:
        print(name)



# 1
program1(100, 999)
# 2
program1(0, 10000)
# 3
program3(5)
# 4
program4(2021)
# 5
program5(0, 10)
# 6
program6(100, 2)
# 7
program7()
