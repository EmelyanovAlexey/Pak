
# Биноминальное распределение
# (N) - кол-во и (P) вероятность успеха
# Лямбда (Экспоненциальное)
# (lambda) - Константа
# Геометрическое
# (P) - Вероятность успеха
# Гипер-геометрическое
# (N) - Всего деталей; (M) - Всего деффектный деталей; (n) - отобранных деталей;
# Равномерное
# (A) - левая граница; (B) - правая граница
# Нормальное Гауса
# (alpha) ??; (sigma) ??;
# Коши
# Пустота, холодна
# Гамма распределение
# (alpha) ??; (lambda) ??;

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from math import factorial
def Cnk(k, n):
    return factorial(n) / (factorial(n-k) * factorial(k))

from math import exp, sqrt, pi
def binominal(returnProbs = False, inputString = ""):
    if inputString != "":
        N, P = inputString.split()
    else:
        N, P = input("Введите N, P через пробел").split()
    N, P = int(N), float(P)
    if P < 0 and P <= 1: raise ValueError("P not correct")
    Q = 1 - P;
    probs = []
    for k in range(N+1):
        probs.append(Cnk(k, N) * P**k * Q**(N-k))
    if returnProbs: return (probs, range(N+1))
    return pd.DataFrame(probs, columns=['P']).plot()

def exponental(returnProbs = False, inputString = ""):
    if inputString != "":
        N, lam = inputString.split()
    else:
        N, lam = input("Введите N, lam через пробел").split()
    N, lam = int(N), float(lam)
    if lam < 0 and lam <= 1: raise ValueError("Lambda not correct")
    probs = []
    for k in range(N+1):
        probs.append(lam * exp(-lam*k))
    if returnProbs: return (probs, range(N+1))
    return pd.DataFrame(probs, columns=['P']).plot()

def geometric(returnProbs = False, inputString = ""):
    if inputString != "":
        N, P = inputString.split()
    else:
        N, P = input("Введите N, P через пробел").split()
    N, P = int(N), float(P)
    Q = 1 - P;
    if P < 0 and P <= 1: raise ValueError("P not correct")
    probs = []
    for k in range(1, N+1):
        probs.append(P*Q**(k-1))
    if returnProbs: return (probs, range(1, N+1))
    return pd.DataFrame(probs, columns=['P']).plot()

def giperGeometric(returnProbs = False, inputString = ""):
    if inputString != "":
        N, M, n = inputString.split()
    else:
        N, M, n = input("Введите N, M, n через пробел").split()
    N, M, n = int(N), int(M), int(n)
    if M > N: raise ValueError("M not correct")
    probs = []
    for m in range(min(M,n)+1):
        probs.append((Cnk(m, M) * Cnk(n-m, N-M))/Cnk(n,N))
    if returnProbs: return (probs, range(min(M,n)+1))
    return pd.DataFrame(probs, columns=['P']).plot()

def ravnomernoe(returnProbs = False, inputString = ""):
    if inputString != "":
        N, A, B = inputString.split()
    else:
        N, A, B = input("Введите N, A, B через пробел").split()
    N, A, B = int(N), int(A), int(B)
    if B <= A: raise ValueError("B is bigger than A")
    probs = [1/(B-A) for x in range(int(A),int(B)+1)]
    if returnProbs: return (probs, range(int(A),int(B)+1))
    return pd.Series(np.array(probs),  index=range(A,B)).plot()

def gause(returnProbs = False, inputString = ""):
    if inputString != "":
        N, alpha, sigma = inputString.split()
    else:
        N, alpha, sigma = input("Введите N, alpha, sigma через пробел").split()
    N, alpha, sigma = int(N), float(alpha), float(sigma)
    probs = []
    for k in range(-N*10, N*10+1):
        probs.append(exp(-((k-alpha)**2)/(2*sigma**2))/(sigma*sqrt(2*pi)))
    if returnProbs: return (probs, [x/10 for x in range(-N*10, N*10+1)])
    return pd.Series(np.array(probs),  index=[x/10 for x in range(-N*10, N*10+1)]).plot()

def gamma(returnProbs = False, inputString = ""):
    if inputString != "":
        N, alpha, lam = inputString.split()
    else:
        N, alpha, lam = input("Введите N, alpha, lam через пробел").split()
    N, alpha, lam = int(N), int(alpha), float(lam)
    if alpha <= 0 or lam <= 0: raise ValueError("alpha or lam below zero")
    probs = []
    for k in range(1, N+1):
        probs.append(
            (lam**alpha * k**(alpha-1) * exp(-lam*k)) / factorial(alpha-1)
            )
    if returnProbs: return (probs, range(1, N+1))
    return pd.Series(np.array(probs)).plot()

def koshi(returnProbs = False, inputString = ""):
    if inputString != "":
        N = int(inputString)
    else:
        N = int(input("Введите N"))
    probs = []
    for k in range(-N, N+1):
        probs.append(
            1/(pi*(1+k**2))
        )
    if returnProbs: return (probs, list(range(-N, N+1)))
    return pd.Series(np.array(probs), index=range(-N, N+1)).plot()





def checkMathEx(N, raspred):
    arr = []
    inputString = input("Введите только параметры (Без N)")
    print(MathEx(raspred, params = str(1) + " " + inputString))
    for x in range(1, N+1):
        probs, inds = raspred(inputString = str(x) + " " + inputString, returnProbs = True)
        arr.append(sum(probs)/len(probs))
    print(arr)
    return pd.Series(arr).plot()

def MathEx(raspred, params, quant = 1):
    probs, inds = raspred(inputString = params, returnProbs = True)
    ex = 0
    for i in range(len(probs)):
        ex += probs[i] * inds[i]**quant
    return ex

def Disp(raspred, params):
    return MathEx(raspred, params, quant=2) - MathEx(raspred, params, quant=1)**2
    
def checkCPT(N, raspred, inputStr = ""):
    arr = []
    if inputStr == "":
        inputString = input("Введите только параметры (Без N)")
    else:
        inputString = inputStr
    Ex = MathEx(raspred, str(1) + " " + inputString)
    Ds = Disp(raspred, str(1) + " " + inputString)
    for x in range(1, N+1):
        params =  str(x) + " " + inputString
        probs, inds = raspred(inputString = params, returnProbs = True)
        arr.append((sum(probs) - x * Ex ) / sqrt(x * Ds))
    return pd.Series(arr).plot()

# gause(inputString="10 0 3")

def gistograma(N, raspred, params = ""):
    arr = [] # N sized
    segments = []
    probs, inds = raspred(returnProbs = True, inputString=params)
    print(probs)
    dlina = len(probs)
    segment = int(dlina / N)
    for i in range(N): #0 1 2
        segments.append((i*segment,(i+1)*segment))
        arr.append(np.trapz(probs[i*segment:(i+1)*segment+1]) / segment)
    return pd.Series(arr, index=segments).plot(kind='bar')

# checkCPT(500, ravnomernoe, "0 500")

gistograma(100, binominal, "5 0.7")

plt.show()


     
