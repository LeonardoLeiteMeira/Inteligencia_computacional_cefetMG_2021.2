from sklearn.preprocessing import PolynomialFeatures

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

def medium_squared_error(y, predicted, m):
    squaredError = np.power(np.subtract(y, predicted), 2)
    return 1/(m) * np.sum(squaredError)

columns = ['size', 'bedrooms', 'price']
df_house_price = pd.read_csv(
    "./data2.txt", sep=",", header=None, names=columns)

column_size = df_house_price['size'].to_numpy()
column_bedrooms =  df_house_price['bedrooms'].to_numpy()

m = len(df_house_price['price'])

matrix_Y =  df_house_price['price'].to_numpy()
matrix_X = np.vstack((column_size, column_bedrooms))
X = np.vstack((np.ones(m), matrix_X)).T

temp1 = np.dot(X.T, X)
temp2 = np.linalg.inv(temp1)
temp3 = np.dot(temp2, X.T)
theta = np.dot(temp3, matrix_Y)

print(theta)

predict_values = [np.dot(theta, X[i]) for i in range(m)]

for i in range(m):
    print(f"{matrix_Y[i]} : {predict_values[i]}")

print(medium_squared_error(matrix_Y, predict_values, m))