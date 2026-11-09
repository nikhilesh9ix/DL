# Week 2 (exam PDF version): Logistic Regression using Gradient Descent (MSE cost, stops at tolerance)
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(y, y_pred):
    return np.mean((y - y_pred) ** 2) / 2

m = int(input("Number of training samples: "))
n = int(input("Number of input features: "))

X = []
y = []
for i in range(m):
    print(f"\nSample {i+1}")
    X.append([float(input(f"X{j+1}: ")) for j in range(n)])
    y.append(int(input("Target (0/1): ")))
X = np.array(X)
y = np.array(y)

weights = np.array([float(input(f"W{i+1}: ")) for i in range(n)])
bias = float(input("Initial bias: "))
learning_rate = float(input("Learning rate: "))
tolerance = float(input("Tolerance: "))

previous_cost = float("inf")
for iteration in range(1, 10001):
    y_pred = sigmoid(X @ weights + bias)
    cost = compute_cost(y, y_pred)

    if abs(previous_cost - cost) < tolerance:
        break
    previous_cost = cost

    error = (y_pred - y) * y_pred * (1 - y_pred)
    dw = (X.T @ error) / m
    db = np.sum(error) / m
    weights -= learning_rate * dw
    bias -= learning_rate * db

y_pred = sigmoid(X @ weights + bias)
cost = compute_cost(y, y_pred)
predicted = (y_pred >= 0.5).astype(int)

print("\nTraining completed")
print("Weights:", np.round(weights, 6))
print("Bias:", round(bias, 6))
print("Cost:", round(cost, 8))
print("Iterations:", iteration)
print("Predicted outputs:", predicted)
