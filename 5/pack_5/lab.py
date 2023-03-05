import numpy as np
import pandas as pd
from tabulate import tabulate

# 1 --------------
# Данные, которые предоставил кинотеатр находятся в файлах cinema_sessions.csv и titanic_with_labels

# Пол (sex): отфильтровать строки, где пол не указан, преобразовать оставшиеся в число 0/1;
# Номер ряда в зале (row_number): заполнить вместо NAN максимальным значением ряда;
# Количество выпитого в литрах (liters_drunk): отфильтровать отрицательные значения и нереально большие значения (выбросы). Вместо них заполнить средним.

seances_df = pd.read_csv('./data/cinema_sessions.csv', sep=' ')
titanic_df = pd.read_csv('./data/titanic_with_labels.csv', sep=' ')

# print(tabulate(seances_df.head(), headers='keys', tablefmt='psql'))
# print(tabulate(titanic_df.head(), headers='keys', tablefmt='psql'))

# sex
# print(titanic_df['sex']) # до
# list_sex_m = ["M", "m", "М", "м"]
# list_sex_f = ["F", "ж", "Ж", "f"]
titanic_df["sex"] = titanic_df["sex"].replace(["F", "ж", "Ж", "f"], 1).replace(["M", "m", "М", "м"], 0)
titanic_df = titanic_df[titanic_df["sex"].apply(str).str.isdigit()]
# print(titanic_df['sex']) # после

# row_number
# запонение нулевых значений (fillna)
titanic_df['row_number'] = titanic_df['row_number'].fillna(titanic_df['row_number'].max())

# liters_drunk
litersDrunk = titanic_df['liters_drunk'][(titanic_df['liters_drunk'] >= 0) & (titanic_df['liters_drunk'] <= 20)]
titanic_df.loc[(titanic_df['liters_drunk'] >= 20) | (titanic_df['liters_drunk'] <= 0), 'liters_drunk'] = int(litersDrunk.mean())
# print(tabulate(titanic_df.head(), headers='keys', tablefmt='psql'))

# 2 ----------------
# 4 Возраст (age): разделить на 3 группы: дети (до 18 лет), взрослые (18 - 50), пожилые (50+). закодировать в виде трёх столбцов с префиксом age_. 
# Старый столбец с age удалить;
# 5 Напиток (drink): преобразовать в число 0/1 был ли этот напиток хмельным;
# 6 Номер чека (check_number): надо сопоставить со второй таблицей со временем сеанса. 
# И закодировать в виде трёх столбцов, был ли это утренний (morining) сеанс, дневной (day) или вечерний (evening).

def selectAgeGroup(age, ageGroups):
    for group in ageGroups:
        if age < group[0]:
            return group[1]
    return None

def selectDrinkType(drink, drinkGroups):
    for group in drinkGroups:
        if drink in group[0]:
            return group[1]
    return None

def selectSessionType(time, timeGroups):
    return selectAgeGroup(time, timeGroups) # :)

#Age's
titanic_df['age'] = titanic_df['age'].apply(selectAgeGroup, ageGroups=[(18, "Child"), (50, "Adult"), (999, "Elderly")])

#Alcohol content of beverages
titanic_df['drink'] = titanic_df['drink'].apply(selectDrinkType, drinkGroups=[(['Beerbeer', 'Bugbeer', 'Strong beer'], 1), (['Cola', 'Fanta', 'Water'], 0)])

#Add session type
titanic_df = pd.merge(titanic_df, seances_df, on="check_number").rename(columns={"session_start" : "session_type"})
titanic_df['session_type'] = titanic_df['session_type'].apply(selectSessionType, timeGroups=[("12:00:00.000", "Morning"), ("17:00:00.000", "Afternoon"), ("24:00:00.000", "Evening")])
print(tabulate(titanic_df.head(), headers='keys', tablefmt='psql'))