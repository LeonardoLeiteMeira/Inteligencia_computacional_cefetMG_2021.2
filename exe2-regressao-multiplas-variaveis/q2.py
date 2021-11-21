import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

num_iterations = 2000
y = []
hypothesis_array = []
theta = []
m = 0
cost_array = []

def normalization_of_x(x, average, standard_deviation):
    return (x-average)/standard_deviation


def normalize_column(column):
    average = np.average(column)
    standard_deviation_value = np.std(column)
    size_of_column = len(column)
    new_array = [normalization_of_x(
        column[index], average, standard_deviation_value) for index in range(0, size_of_column)]
    return np.array(new_array)


def normalize_df(df):
    new_size_column = normalize_column(df['size'])
    new_bedrooms_column = normalize_column(df['bedrooms'])
    new_price_column = normalize_column(df['price'])
    new_df = pd.DataFrame(
        {'size': new_size_column, 'bedrooms': new_bedrooms_column, 'price': new_price_column})
    new_df.columns = ['size', 'bedrooms', 'price']
    return new_df


def cost_function(X, y, theta):
    m = len(y)
    hypothesis = np.sum(np.multiply(X, theta), axis=1)
    squaredError = np.power(np.subtract(hypothesis, y), 2)
    cost = 1/(2*m) * np.sum(squaredError)
    return cost


def gradient_descendent(X, y, theta, alpha):
    m = len(y)

    hypothesis = np.sum(np.multiply(X, theta), axis=1)
    error = np.subtract(hypothesis, y)
    theta_new = alpha * 1/m * np.sum(np.multiply(X.T, error), axis=1)
    theta = np.subtract(theta, theta_new)

    return theta


def execute(alpha):
    global theta, hypothesis_array, y, m, cost_array
    m = len(normalized_df['price'])
    theta0 = random.randint(1, 25)
    theta1 = random.randint(1, 25)
    theta2 = random.randint(1, 25)

    theta = np.asarray([theta0, theta1, theta2]).astype(float)
    x1 = normalized_df['size'].to_numpy()
    x2 = normalized_df['bedrooms'].to_numpy()
    X = np.vstack((x1, x2))
    X = np.vstack((np.ones(m), X)).T
    y = normalized_df['price'].to_numpy()
    cost_array = []
    hypothesis_array = []

    for i in range(num_iterations):
        hypothesis_array.append(np.sum(np.multiply(X, theta), axis=1))
        theta = gradient_descendent(X, y, theta, alpha)
        cost_array.append(cost_function(X, y, theta))

    return cost_array


columns = ['size', 'bedrooms', 'price']
df_house_price = pd.read_csv(
    "./data2.txt", sep=",", header=None, names=columns)

normalized_df = normalize_df(df_house_price)

print(normalized_df)


alpha = [0.001, 0.01, 0.05, 0.1, 1]
colors = ['blue', 'red', 'green', 'yellow', 'purple']

for i in range(0, len(alpha)):
    plt.plot(execute(alpha[i]), color=colors[i],
             linewidth=1, label=f"alpha={alpha[i]}")

plt.xlabel("iteracoes")
plt.ylabel("custo")
plt.legend()
plt.show()

print(theta)
print(cost_array[m])
# -----------------------------------------------------------

# 2.3 -> Não é possivel traçar o ajuste linear, pois agora temos multiplas dimensões
