from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as nrand

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# тестовые данные -

# 1 Разделите данные Титаника (train.csv) на тренировочную, валидационную и тестовую часть.
# С помощью валидационной части подберите гиперпараметры для моделей Random Forest, XGBoost, Logistic Regression и KNN.
# Получите точность этих моделей на тестовой части.

# 2 С помощью RandomForest выберите 2, 4, 8 самых важных признаков и проверьте точность моделей только на этих признаках.

# 3 Используя координаты скважин из файла wells_info.csv разделите их на кластера с помощью любых 4 методов и отобразите разделение.
# Параметры подбираются самостоятельно.

# 4 Приведите отобранные в 6.1 задании признаки из файла wells_info_with_prod.csv в двумерное пространство. Выделите цветом добычу с этой скважины.


# 1 -----------------------------------------------------------------------------
# Разделите данные Титаника (train.csv) на тренировочную, валидационную и тестовую часть.
# С помощью валидационной части подберите гиперпараметры для моделей Random Forest, XGBoost, Logistic Regression и KNN.
# Получите точность этих моделей на тестовой части.

# functions
# переводим значения на 0 и 1
def prepare_num(df):
    df_num = df.drop(['Sex', 'Embarked', 'Pclass'], axis=1)
    df_sex = pd.get_dummies(df['Sex'])
    df_emb = pd.get_dummies(df['Embarked'], prefix='Emb')
    df_pcl = pd.get_dummies(df['Pclass'], prefix='Pclass')
    df_num = pd.concat((df_num, df_sex, df_emb, df_pcl), axis=1)
    return df_num

# ----------------------------- 1


# 1 - считали данные
full_data_titanic = pd.read_csv('./data/train.csv')
# print(full_data_titanic.head())

# 2 - убрали не нужные нам колонки
df_full_titanic = full_data_titanic.drop(
    ['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
# print(df_full_titanic.head())

# 3 - переводим значения на 0 и 1
train_full_df = prepare_num(df_full_titanic)
# print(train_full_df.head())

# 4 - нужно заменить значения null на какие-то определенные
# median позволяет выбрать ср.зн по колонке, mean
train_full_df = train_full_df.fillna(train_full_df.median())
# print(train_full_df.head())

# данные подготовлены

# 5 - разбиваем данные на train valid test
# train - Обучающий набор: это подмножество данных, которые я буду использовать для обучения модели.
# valid - Валидационная выборка: используется для контроля процесса обучения.
# Она поможет предотвратить переобучение и обеспечит более точную настройку входных параметров.
# test - Тестовый набор: подмножество данных для оценки производительности модели.

# Чтобы принять решение о размере каждого подмножества,
# по правилу 80-20 (80 % для тренировочного сплита, 20% для тестового сплита)
# или по правилу 70-20-10 (70% для обучения, 20% для проверки, 10% для тестирования) и т.д.

# максимальное абсолютное значение каждой функции масштабировалось до размера единицы. Этого можно добиться с помощью MinMaxScaler
X = MinMaxScaler().fit_transform(train_full_df)
Y = full_data_titanic.Survived

X_train, X_valid, Y_train, Y_valid = train_test_split(
    X, Y, test_size=0.25, random_state=42)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)


# 6 - обучение
# Random Forest, XGBoost, Logistic Regression и KNN
iteration = 100

# Random Forest - случайный лес

better_accuracy_valid = 0
better_param = ''
better_accuracy_test = 0
best_model = 0
criterionArr = ["gini", "entropy", "log_loss"]

for i in range(iteration):
    n_estimators = nrand.randint(1, 30)  # Количество деревьев в лесу
    max_depth = nrand.randint(1, 10)  # Максимальная глубина дерева
    criterion = criterionArr[nrand.randint(0, 3)]

    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, criterion=criterion)
    model.fit(X_train, Y_train)
    predict = model.predict(X_valid)
    score = model.score(X_valid, Y_valid)
    if (score > better_accuracy_valid):
        best_model = model
        better_accuracy_valid = score
        better_param = ' n_estimators=' + \
            str(n_estimators) + ' | max_depth=' + \
            str(max_depth) + ' | criterion=' + criterion
        predict = model.predict(X_test)
        better_accuracy_test = model.score(X_test, Y_test)

print('Random Forest ----------------------------------------------------------------')
print("Лучшая точность valid:  ", str(
    round(better_accuracy_valid, 4)*100) + "%")
print("Лучшая параметры :  ", better_param)
print("Лучшая точность test:  ", str(round(better_accuracy_test, 4)*100) + "%")

# сразу определяем задание 2
# С помощью RandomForest выберите 2, 4, 8 самых важных признаков и проверьте точность моделей только на этих признаках
# ....
print('Random Forest second task ----------------------------------------------------')

importanceRandomForest = best_model.feature_importances_
features = train_full_df.columns
indices = np.argsort(importanceRandomForest)
plt.barh(range(len(indices)),
         importanceRandomForest[indices], color='g',)
plt.yticks(range(len(indices)), features[indices])
plt.show()

bestTags = []
for cnt in range(2, 9, 2):
    bestTags.append(list(features[indices][::-1][:cnt]))

print("\n////////////////////////////////////////////////////////////")
for tags in bestTags:
    # такие параметры мне выдал выше
    modelRF = RandomForestClassifier(
        n_estimators=13, max_depth=9, criterion='entropy')
    usedDataScaled = MinMaxScaler().fit_transform(train_full_df[tags])
    X_train_2, X_test_2, Y_train_2, Y_test_2 = train_test_split(
        usedDataScaled, Y, test_size=0.2, random_state=42)
    modelRF.fit(X_train_2, Y_train_2)
    res_model = modelRF.score(X_test_2, Y_test_2)
    print(f"Признаки: {tags}")
    print('Точность: ' + str(round(res_model, 4)*100) + "%")
    print("////////////////////////////////////////////////////////////\n")

# XGBoost

better_accuracy_valid = 0
better_param = ''
better_accuracy_test = 0

for i in range(iteration):
    n_estimators = nrand.randint(1, 30)
    max_depth = nrand.randint(1, 10)

    model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, Y_train)
    predict = model.predict(X_valid)
    score = model.score(X_valid, Y_valid)
    if (score > better_accuracy_valid):
        better_accuracy_valid = score
        better_param = ' n_estimators=' + \
            str(n_estimators) + ' | max_depth=' + str(max_depth)
        predict = model.predict(X_test)
        better_accuracy_test = model.score(X_test, Y_test)

print('xgboost ----------------------------------------------------------------')
print("Лучшая точность valid:  ", str(
    round(better_accuracy_valid, 4)*100) + "%")
print("Лучшая параметры :  ", better_param)
print("Лучшая точность test:  ", str(round(better_accuracy_test, 4)*100) + "%")

# Logistic

better_accuracy_valid = 0
better_param = ''
better_accuracy_test = 0
solverArr = ['lbfgs', 'liblinear', 'newton-cg',
             'newton-cholesky', 'sag', 'saga']

for i in range(iteration):
    C = nrand.randint(1, 30)/100
    solver = solverArr[nrand.randint(0, 6)]

    model = LogisticRegression(C=C, solver=solver)
    model.fit(X_train, Y_train)
    predict = model.predict(X_valid)
    score = model.score(X_valid, Y_valid)
    if (score > better_accuracy_valid):
        better_accuracy_valid = score
        better_param = ' C=' + \
            str(C) + ' | solver=' + solver
        predict = model.predict(X_test)
        better_accuracy_test = model.score(X_test, Y_test)

print('Logistic ----------------------------------------------------------------')
print("Лучшая точность valid:  ", str(
    round(better_accuracy_valid, 4)*100) + "%")
print("Лучшая параметры :  ", better_param)
print("Лучшая точность test:  ", str(round(better_accuracy_test, 4)*100) + "%")

# KNN - Метод ближайших соседей

better_accuracy_valid = 0
better_accuracy_test = 0

for i in range(iteration):
    model = KNeighborsClassifier()
    model.fit(X_train, Y_train)
    predict = model.predict(X_valid)
    score = model.score(X_valid, Y_valid)
    if (score > better_accuracy_valid):
        better_accuracy_valid = score
        predict = model.predict(X_test)
        better_accuracy_test = model.score(X_test, Y_test)

print('KNN ----------------------------------------------------------------')
print("Лучшая точность valid:  ", str(
    round(better_accuracy_valid, 4)*100) + "%")
print("Лучшая точность test:  ", str(round(better_accuracy_test, 4)*100) + "%")


# 3 Используя координаты скважин из файла wells_info.csv разделите их на кластера с помощью любых 4 методов и отобразите разделение.
# Параметры подбираются самостоятельно.

wells_data = pd.read_csv("data/wells_info.csv")
wells_info_drop = np.array(
    wells_data[["BottomHoleLatitude", "BottomHoleLongitude"]])

model = KMeans(n_clusters=5, n_init='auto')
clusters = model.fit_predict(wells_info_drop)

for cl in clusters:
    dataCl = wells_info_drop[clusters == cl]
    plt.scatter(dataCl[:, 0], dataCl[:, 1])
plt.show()


# 4 Приведите отобранные в 6.1 задании признаки из файла wells_info_with_prod.csv в двумерное пространство.
# Выделите цветом добычу с этой скважины.
wells_info_with_prod = pd.read_csv("data/wells_info_with_prod.csv")
print(wells_info_with_prod.head())
# не понял что делать

# мои результаты
# 1
# Random Forest ----------------------------------------------------------------
# Лучшая точность valid:   86.1%
# Лучшая параметры :    n_estimators=7 | max_depth=9 | criterion=gini
# Лучшая точность test:   84.36%
# xgboost ----------------------------------------------------------------
# Лучшая точность valid:   85.2%
# Лучшая параметры :    n_estimators=19 | max_depth=6
# Лучшая точность test:   84.92%
# Logistic ----------------------------------------------------------------
# Лучшая точность valid:   81.17%
# Лучшая параметры :    n_estimators=27 | max_depth=2 | criterion=log_loss
# Лучшая точность test:   79.89%
# KNN ----------------------------------------------------------------
# Лучшая точность valid:   82.06%
# Лучшая точность test:   80.45%

# 2
# Random Forest second task ----------------------------------------------------
# Признаки: ['male', 'Fare']
# Точность: 80.45%
# ////////////////////////////////////////////////////////////
# Признаки: ['male', 'Fare', 'female', 'Age']
# Точность: 76.53999999999999%
# ////////////////////////////////////////////////////////////
# Признаки: ['male', 'Fare', 'female', 'Age', 'Pclass_3', 'Parch']
# Точность: 81.01%
# ////////////////////////////////////////////////////////////
# Признаки: ['male', 'Fare', 'female', 'Age', 'Pclass_3', 'Parch', 'Pclass_1', 'SibSp']
# Точность: 79.89%
# ////////////////////////////////////////////////////////////

# 3 там график
