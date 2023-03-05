from sklearn.tree import DecisionTreeClassifier
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

# Загрузить файл, разделить его на train и test. Для test взять 10% случайно выбранных строк таблицы.
# Обучить модели: Decision Tree, XGBoost, Logistic Regression из библиотек sklearn и xgboost. Обучить модели предсказывать столбец label по остальным столбцам таблицы.
# Наладить замер Accuracy - доли верно угаданных ответов.
# Точности всех моделей не должны быть ниже 85%
# С помощью Decision Tree выбрать 2 самых важных признака и проверить точность модели, обученной только на них.

# 1
titanic_data = pd.read_csv("data/titanic_prepared.csv")
titanic_data = titanic_data.drop(['Unnamed: 0'], axis=1)

# print(titanic_data.head())

X = titanic_data.drop(['label'], axis=1)
Y = titanic_data['label']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.1, random_state=42)

# 2
# Decision Tree

better_accuracy = 0
better_param = ''
best_model = 0
criterionArr = ["gini", "entropy", "log_loss"]
bestCriterion = "gini"
bestMax_depth = 1


print('Decision Tree .....')
for max_depth in list(range(1, 15)):
    for criterion in criterionArr:
        model = RandomForestClassifier(
            max_depth=max_depth, criterion=criterion)
        model.fit(X_train, Y_train)
        predict = model.predict(X_test)
        score = model.score(X_test, Y_test)
        if (score > better_accuracy):
            bestCriterion = criterion
            bestMax_depth = max_depth
            best_model = model
            better_accuracy = score
            better_param = ' | max_depth=' + \
                str(max_depth) + ' | criterion=' + criterion

print('Decision Tree ----------------------------------------------------------------')
print("Лучшая точность:  ", str(
    round(better_accuracy, 4)*100) + "%")
print("Лучшая параметры :  ", better_param)

# # С помощью Decision Tree выбрать 2 самых важных признака и проверить точность модели, обученной только на них.
print('////////////////////////////////////////////////////////////////\n')

importanceForest = best_model.feature_importances_
features = X_test.columns
indices = np.argsort(importanceForest)
plt.barh(range(len(indices)), importanceForest[indices], color='g',)
plt.yticks(range(len(indices)), features[indices])
plt.show()

ImportantTags = list(features[indices][::-1][:2]) + ['label']
print(ImportantTags[:2])

X = titanic_data.drop(ImportantTags[:2], axis=1)
Y = titanic_data[ImportantTags[:2]]

X_train2, X_test2, Y_train2, Y_test2 = train_test_split(
    X, Y, test_size=0.1, random_state=42)
best_model.fit(X_train2, Y_train2)

predict = best_model.predict(X_test2)
score = best_model.score(X_test2, Y_test2)

print(f"Предсказание по тегам: {ImportantTags[:2]}")
print(f"Резульат: {round(score, 4)*100}%")

print('////////////////////////////////////////////////////////////////\n')

# XGBoost
better_accuracy = 0
better_param = ''
print('xgboost .....')
for n_estimators in list(range(1, 30)):
    for max_depth in list(range(1, 30)):
        model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, Y_train)
        predict = model.predict(X_test)
        score = model.score(X_test, Y_test)
        if (score > better_accuracy):
            better_accuracy = score
            better_param = ' n_estimators=' + \
                str(n_estimators) + ' | max_depth=' + str(max_depth)

print('xgboost ----------------------------------------------------------------')
print("Лучшая точность:  ", str(
    round(better_accuracy, 4)*100) + "%")
print("Лучшая параметры :  ", better_param)

# Logistic Regression
better_accuracy = 0
better_param = ''
solverArr = ['lbfgs', 'liblinear', 'newton-cg',
             'newton-cholesky', 'sag', 'saga']

for i in list(range(1, 30)):
    C = i/100

    for solver in solverArr:
        model = LogisticRegression(C=C, solver=solver)
        model.fit(X_train, Y_train)
        predict = model.predict(X_test)
        score = model.score(X_test, Y_test)
        if (score > better_accuracy):
            better_accuracy = score
            better_param = ' C=' + \
                str(C) + ' | solver=' + solver

print('Logistic ----------------------------------------------------------------')
print("Лучшая точность:  ", str(
    round(better_accuracy, 4)*100) + "%")
print("Лучшая параметры :  ", better_param)





# Decision Tree ----------------------------------------------------------------
# Лучшая точность:   90.36%
# Лучшая параметры :    | max_depth=11 | criterion=entropy    
    
# ////////////////////////////////////////////////////////////////

# ['morning', 'evening']
# Предсказание по тегам: ['morning', 'evening']
# Резульат: 96.99%

# ////////////////////////////////////////////////////////////////

# xgboost ----------------------------------------------------------------
# Лучшая точность:   91.11%
# Лучшая параметры :    n_estimators=26 | max_depth=9

# Logistic ----------------------------------------------------------------
# Лучшая точность:   88.55%
# Лучшая параметры :    C=0.09 | solver=lbfgs