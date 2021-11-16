import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

num_iterations = 2000
alpha = 0.01

def hypothesis_model(theta0, theta1, x):
    return theta0 + (theta1 * x)

def error(hx, y):
    errors = []
    for i in range(len(hx)):
        errors.append((hx[i] - y[i])**2)
    
    return errors

def multiply_arrays(a, b):
    result = []
    for i in range(len(a)):
        result.append(a[i]*b[i])
    
    return result

def cost_function(m, hx, y):
    return 1/(2*m) * sum_positions(error(hx, y))

def sum_positions(array):
    sum = 0
    for i in range(0,len(array)):
        sum = sum + array[i]
    return sum

def cost_function(x, y, theta):
    m = len(y)
    hypothesis = np.sum(np.multiply(X, theta), axis=1)
    squaredError = np.power(np.subtract(hypothesis, y), 2)
    cost = 1/(2*m) * np.sum(squaredError)
    return cost

def gradient_descendent(X, y, theta, alpha, num_iterations):
    m = len(y)

    hypothesis = np.sum(np.multiply(X, theta), axis=1)
    error = np.subtract(hypothesis, y)
    theta_new = alpha * 1/m * np.sum(np.multiply(X.T, error), axis=1)
    theta = np.subtract(theta, theta_new)
    
    return theta

df_city_profit = pd.read_csv("./data1.txt", sep=",", header=None)
df_city_profit.columns = ['population', 'profit']
print(df_city_profit)

# df_city_profit.plot.scatter(x="population", y="profit")

m = len(df_city_profit.profit)
theta0 = random.randint(1, 25)
theta1 = random.randint(1, 25)

theta = np.asarray([theta0,theta1]).astype(float)
x = df_city_profit.population.to_numpy()
X = np.vstack((np.ones(m), x.T)).T
y = df_city_profit.profit.to_numpy()
cost_array = []
hypothesis_array = []


for i in range(num_iterations):
    cost = cost_function(X, y, theta)
    cost_array.append(cost)
    hypothesis_array.append(np.sum(np.multiply(X, theta), axis=1))
    theta = gradient_descendent(X, y, theta, alpha, num_iterations)

# plt.plot([i for i in range(0, num_iterations)], cost_array)

df_city_profit.plot.scatter(x="population", y="profit")
plt.plot(X[:,[1]], np.sum(np.multiply(X,theta), axis=1), color='red', linewidth=1)
plt.show()
