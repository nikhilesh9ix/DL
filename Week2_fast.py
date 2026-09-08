import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost(y, p):
    return np.mean((y - p) ** 2) / 2

m = int(input("Samples: "))
n = int(input("Features: "))

X = []
Y = []

for i in range(m):
    X.append([float(input(f"X{j+1}: ")) for j in range(n)])
    Y.append(int(input("Target (0/1): ")))

X = np.array(X)
Y = np.array(Y)

W = np.array([float(input(f"W{i+1}: ")) for i in range(n)])
b = float(input("Bias: "))
lr = float(input("Learning rate: "))
tol = float(input("Tolerance: "))

old_cost = float("inf")

for i in range(10000):
    Z = np.dot(X, W) + b
    P = sigmoid(Z)

    error = (P - Y) * P * (1 - P)
    dW = np.dot(X.T, error) / m
    db = np.mean(error)

    W -= lr * dW
    b -= lr * db

    new_cost = cost(Y, P)

    if abs(old_cost - new_cost) < tol:
        break
    old_cost = new_cost

P = sigmoid(np.dot(X, W) + b)
prediction = (P >= 0.5).astype(int)

print("\nTraining Completed")
print("Weights:", W)
print("Bias:", b)
print("Cost:", cost(Y, P))
print("Predicted:", prediction)