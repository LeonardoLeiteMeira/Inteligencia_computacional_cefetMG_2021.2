import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

num_iterations = 2000
alpha = 0.01


def normalization_of_x(x, average, standard_deviation):
    return (x-average)/standard_deviation

def normalize_column(column):
    average = np.average(column)
    standard_deviation_value = np.std(column)
    size_of_column = len(column)
    new_array = [ normalization_of_x(column[index],average,standard_deviation_value) for index in range(0,size_of_column) ]
    return np.array(new_array)

def normalize_df(df):
    new_size_column = normalize_column(df['size'])
    new_bedrooms_column = normalize_column(df['bedrooms'])
    new_price_column = normalize_column(df['price'])
    new_df = pd.DataFrame({'size': new_size_column, 'bedrooms' : new_bedrooms_column, 'price': new_price_column })
    new_df.columns = ['size', 'bedrooms', 'price']
    return new_df

columns = ['size', 'bedrooms', 'price']
df_house_price = pd.read_csv("./data2.txt", sep=",", header=None, names=columns)
print(df_house_price)

print("-----------")

print(normalize_df(df_house_price))







