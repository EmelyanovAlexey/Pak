import numpy as np
import pandas as pd

from dateutil.relativedelta import relativedelta
from statistics import mode

# 1 ---------------
# Создайте DataFrame с 5 столбцами и 10 строками, заполненный случайными числами от 0 до 1. 
# По каждой строке посчитайте среднее чисел, которые больше 0.3.

myRow = 10
myColumn = 5

data = np.random.random(size=(myRow, myColumn))
column_names = [chr(x) for x in range(ord('a'), ord('a')+data.shape[1])]
# По каждой строке посчитайте среднее чисел, которые больше 0.3.
averages = [np.mean(x, where=[m > 0.3 for m in x]) for x in data]
data = np.column_stack((data, averages))
df = pd.DataFrame(data, columns=column_names + ['averages'])
df
# print(df)


# 2 ---------------
# Посчитайте, сколько целых месяцев длилась добыча на каждой скважине в файле wells_info.csv.

# возвращает длительность добычи в скважинах 
def get_months_extraction(df):
    # приобразование в дата формат
    startProd, endProd = pd.to_datetime(df[['FirstProductionDate', 'CompletionDate']])
    df['FullMonthsFromStartToEnd'] = relativedelta(endProd, startProd).months

    return df

df = pd.read_csv('data/wells_info.csv')
# print(df.head())
df.insert(5, 'FullMonthsFromStartToEnd', '')
# print(df)
df.apply(get_months_extraction, axis=1)
# print(df)


# 3 ---------------
# Заполните пропущенные числовые значения медианой, а остальные самым часто встречаемым значением в файле wells_info_na.csv.

df = pd.read_csv('data/wells_info.csv')
# print(df.head())

def write_param(column):
    if column.dtype == np.float64:
        swap_val = np.nanmedian(column) # заполняю медианой
    else:
        no_nan = list(filter(lambda col: col==col, column))
        swap_val = mode(no_nan) # получить статистику часто встречаемых
    
    return column.fillna(swap_val)

df.apply(write_param)
# print(df.head())

# 4 ---------------
# Используя файл production.csv добавьте к каждой скважине в файле wells_info.csv 
# колонку с информацией о суммарной добыче за всё время и за первые 12 месяцев.

wells_info = pd.read_csv("data/wells_info.csv")
prod_info  = pd.read_csv("data/production.csv")

def get_sum(df):
    # суммы в колонках
    total_liq = sum(df[:12]["Liquid"])
    total_gas = sum(df[:12]["Gas"])
    out = pd.Series(data=[total_liq,total_gas], index=['Liquid 12 m', 'Gas 12 m'], dtype=object)

    return out

total_prod_by_api = prod_info[["API", "Liquid", "Gas"]].groupby("API").sum()
m12_prod_by_api   = prod_info[["API", "Liquid", "Gas"]].groupby("API").apply(get_sum)
res = pd.merge(wells_info, pd.merge(total_prod_by_api, m12_prod_by_api, on='API'), on='API')

print(res)




