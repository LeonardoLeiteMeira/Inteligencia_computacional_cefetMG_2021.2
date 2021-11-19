import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

num_iterations = 2000
alpha = 0.01

def hypothesis_model(theta0, theta1, x):
    return theta0 + (theta1 * x)

def cost_function(X, y, theta):
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
    hypothesis_array.append(np.sum(np.multiply(X, theta), axis=1))
    theta = gradient_descendent(X, y, theta, alpha, num_iterations)
    cost_array.append(cost_function(X, y, theta))

plt.plot(cost_array[0:len(cost_array)],color='blue', linewidth=1)
plt.show()

df_city_profit.plot.scatter(x="population", y="profit")
plt.plot(X[:,[1]], np.sum(np.multiply(X,theta), axis=1), color='red', linewidth=1)
plt.show()
