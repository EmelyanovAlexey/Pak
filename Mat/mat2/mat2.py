import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from math import factorial, exp, sqrt, pi

listName = ["1 Биноминальное распределение (N) - кол-во и (P) вероятность успеха",
            "2 Лямбда (Экспоненциальное) (lambda) - Константа",
            "3 Геометрическое (P) - Вероятность успеха",
            "4 Гипер-геометрическое (N) - Всего деталей; (M) - Всего деффектный деталей; (n) - отобранных деталей;",
            "5 Равномерное (A) - левая граница; (B) - правая граница",
            "6 Нормальное Гауса (alpha) ??; (sigma) ??;",
            "7 Коши Пустота, холодна",
            "8 Выход"]

def Cnk(k, n):
    return factorial(n) / (factorial(n-k) * factorial(k))

def renderPlot(points = [], size=100, typePlot="bar"):
    pointsArr = []
    segments = []
    segment = int(len(points) / size)
    for i in range(size):
        segments.append((i*segment,(i+1)*segment))
        pointsArr.append(np.trapz(points[i*segment:(i+1)*segment+1]) / segment)
    pd.Series(pointsArr, index=segments).plot(kind=typePlot)
    plt.show()
    
# -----------------------------------------
def menuPlotType():
    listPlotType = ["bar", "area", "line"]
    cnt = 1;
    for i in listPlotType:
        print((cnt), " - "+ i)
        cnt+=1
    type = int(input("Выбор типа: "))
    if (type-1 < 0 and type > 3):
        menuPlotType()
    return listPlotType[type-1]

# Биноминальное распределение
def binominal():
    res = []
    # ввод данных
    n = int(input("Введите n: "))
    p = float(input("Введите p: "))
    size = int(input("Введите кол-во отрезков на которые делим гистограмму size: "))
    typePlotEnter = menuPlotType()
    q = 1 - p
    # поиск
    for k in range(int(n)+1):
        res.append(Cnk(k, n) * p**k * q**(n-k))
    # отрисовка
    renderPlot(points = res, size=size, typePlot=typePlotEnter)

# Лямбда (Экспоненциальное)
def exponental():
    res = []
    n = int(input("Введите n: "))
    lam = float(input("Введите lam: "))
    size = int(input("Введите кол-во отрезков на которые делим гистограмму size: "))
    typePlotEnter = menuPlotType()
    # поиск
    for k in range(n+1):
        res.append(lam * exp(-lam*k))
    # отрисовка
    renderPlot(points = res, size=size, typePlot=typePlotEnter)

# Геометрическое
def geometric():
    res = []
    n = int(input("Введите n: "))
    p = float(input("Введите p: "))
    size = int(input("Введите кол-во отрезков на которые делим гистограмму size: "))
    typePlotEnter = menuPlotType()
    q = 1 - p;
    for k in range(1, n+1):
        res.append(p*q**(k-1))
    # отрисовка
    renderPlot(points = res, size=size, typePlot=typePlotEnter)

# Гипер-геометрическое
def giperGeometric():
    res = []
    N = int(input("Введите N: "))
    M = float(input("Введите параметр расположения M: "))
    n = float(input("Введите параметр расположения n: "))
    size = int(input("Введите кол-во отрезков на которые делим гистограмму size: "))
    typePlotEnter = menuPlotType()

    for t in range(min(M,n)+1):
        res.append((Cnk(t, M) * Cnk(n-t, N-M))/Cnk(n,N))
    # отрисовка
    renderPlot(points = res, size=size, typePlot=typePlotEnter)

# Равномерное
def ravnomernoe():
    res = []
    A = float(input("Введите A: "))
    B = float(input("Введите B: "))
    size = int(input("Введите кол-во отрезков на которые делим гистограмму size: "))
    typePlotEnter = menuPlotType()
    if B <= A: ravnomernoe()
    res = [1/(B-A) for x in range(int(A),int(B)+1)]
    # отрисовка
    renderPlot(points = res, size=size, typePlot=typePlotEnter)

# Нормальное Гауса
def gause():
    res = []
    n = int(input("Введите n: "))
    alpha = float(input("Введите alpha: "))
    sigma = float(input("Введите sigma: "))
    size = int(input("Введите кол-во отрезков на которые делим гистограмму size: "))
    typePlotEnter = menuPlotType()
    
    for k in range(-n*10, n*10+1):
        res.append(exp(-((k-alpha)**2)/(2*sigma**2))/(sigma*sqrt(2*pi)))
    
    # отрисовка
    renderPlot(points = res, size=size, typePlot=typePlotEnter)

# Коши
def koshi():
    res = []
    n = int(input("Введите n: "))
    size = int(input("Введите кол-во отрезков на которые делим гистограмму size: "))
    typePlotEnter = menuPlotType()
    for k in range(-n, n+1):
        res.append(1/(pi*(1+k**2)))
    # отрисовка
    renderPlot(points = res, size=size, typePlot=typePlotEnter)
# --------------------------------------

def checkMathEx():
    res = []
    n = int(input("Введите n "))
    strX = input("Введите последовательность через пробел ")
    x = strX.split(" ")
    typePlotEnter = menuPlotType()
    for i in range(1, n+1):
        Mysum = 0
        for i in x:
            Mysum += int(i)
        res.append(Mysum/len(strX))
    renderPlot(points = res, size=100, typePlot=typePlotEnter)

def MathEx(raspred, params, quant = 1):
    probs, inds = raspred(inputString = params, returnProbs = True)
    ex = 0
    for i in range(len(probs)):
        ex += probs[i] * inds[i]**quant
    return ex

# def Disp(raspred, params):
#     return MathEx("1 2 5 5 4 5", params, quant=2) - MathEx("1 1 1 1 1", params, quant=1)**2


def startMainMenu():
    MainMenu = """===================================================================================================
    =    """ + listName[0] + """
    =    """ + listName[1] + """
    =    """ + listName[2] + """
    =    """ + listName[3] + """
    =    """ + listName[4] + """
    =    """ + listName[5] + """
    =    """ + listName[6] + """
    =    """ + listName[7] + """
===================================================================================================
    """
    number = input(MainMenu + "enter item menu: ")
    if (int(number) == 1):
        os.system('clear')
        print("Ваш выбор ", listName[int(number)-1])
        binominal();
        startMainMenu();
    elif (int(number) == 2):
        os.system('clear')
        print("Ваш выбор ", listName[int(number)-1])
        exponental();
        startMainMenu();
    elif (int(number) == 3):
        os.system('clear')
        print("Ваш выбор ", listName[int(number)-1])
        geometric();
        startMainMenu();
    elif (int(number) == 4):
        os.system('clear')
        print("Ваш выбор ", listName[int(number)-1])
        giperGeometric();
        startMainMenu();
    elif (int(number) == 5):
        os.system('clear')
        print("Ваш выбор ", listName[int(number)-1])
        ravnomernoe();
        startMainMenu();
    elif (int(number) == 6):
        os.system('clear')
        print("Ваш выбор ", listName[int(number)-1])
        gause();
        startMainMenu();
    elif (int(number) == 7):
        os.system('clear')
        print("Ваш выбор ", listName[int(number)-1])
        koshi();
        startMainMenu();
    elif (int(number) == 8):
        os.system('clear')
        print("Пока :)")
    else:
        os.system('clear')
        startMainMenu()

startMainMenu()