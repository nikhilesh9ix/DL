# Week 2: Neural Network for Logistic Regression (minimize cost, update parameters)
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

m = int(input("Samples: "))
n = int(input("Features: "))

X = []
for i in range(m):
    X.append([float(input(f"X{j+1}: ")) for j in range(n)])
X = np.array(X)
Y = np.array([int(input("Target: ")) for i in range(m)])

W = np.array([float(input(f"Weight {i+1}: ")) for i in range(n)])
b = float(input("Bias: "))
lr = float(input("Learning rate: "))

for i in range(10000):
    P = sigmoid(X @ W + b)
    error = (P - Y) * P * (1 - P)
    dW = X.T @ error / m
    db = np.mean(error)
    W -= lr * dW
    b -= lr * db

P = sigmoid(X @ W + b)

print("Weights:", W)
print("Bias:", b)
print("Cost:", np.mean((Y - P) ** 2) / 2)
print("Prediction:", (P >= 0.5).astype(int))
