# Week 3: Two-Class Classification using Tanh and Cross-Entropy
# (keeps the Week 3 observation logic as written)
import numpy as np

def compute_cost(y, y_pred):
    return -np.mean(y * np.log(y_pred + 1e-10) + (1 - y) * np.log(1 - y_pred + 1e-10))

m = int(input("Number of training samples: "))
n = int(input("Number of input features: "))

X, y = [], []
for i in range(m):
    print(f"\nSample {i+1}")
    X.append([float(input(f"X{j+1}: ")) for j in range(n)])
    y.append(int(input("Target (0/1): ")))
X, y = np.array(X), np.array(y)

weights = np.array([float(input(f"W{i+1}: ")) for i in range(n)])
bias = float(input("Initial bias: "))
learning_rate = float(input("Learning rate: "))
tolerance = float(input("Tolerance: "))

previous_cost = float("inf")
for iteration in range(1, 10001):
    y_pred = np.tanh(X @ weights + bias)
    cost = compute_cost(y, y_pred)

    error = (y_pred - y) * y_pred * (1 - y_pred)
    weights -= learning_rate * ((X.T @ error) / m)
    bias -= learning_rate * (np.sum(error) / m)

    if abs(previous_cost - cost) < tolerance:
        break
    previous_cost = cost

print("\nTraining completed")
print("Weights:", np.round(weights, 6))
print("Bias:", round(bias, 6))
print("Cost:", round(cost, 8))
print("Iterations:", iteration)
print("Predicted outputs:", (y_pred >= 0.5).astype(int))
