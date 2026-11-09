# Week 4: Deep Neural Network with Two Hidden Layers
import numpy as np

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

m = int(input("Samples: "))
n = int(input("Features: "))

X = np.array([[float(input(f"X{j+1}: ")) for j in range(n)] for i in range(m)])
Y = np.array([int(input("Target: ")) for i in range(m)]).reshape(-1, 1)

h1 = int(input("Hidden layer 1: "))
h2 = int(input("Hidden layer 2: "))
lr = float(input("Learning rate: "))

np.random.seed(1)
W1 = np.random.randn(n, h1) * 0.1
W2 = np.random.randn(h1, h2) * 0.1
W3 = np.random.randn(h2, 1) * 0.1
b1 = np.ones((1, h1)) * 0.1
b2 = np.ones((1, h2)) * 0.1
b3 = 0.1

for i in range(10000):
    # Forward
    A1 = relu(X @ W1 + b1)
    A2 = relu(A1 @ W2 + b2)
    P = sigmoid(A2 @ W3 + b3)

    # Backward
    D3 = P - Y
    D2 = (D3 @ W3.T) * (A2 > 0)
    D1 = (D2 @ W2.T) * (A1 > 0)

    # Update
    W3 -= lr * (A2.T @ D3) / m
    b3 -= lr * np.mean(D3)
    W2 -= lr * (A1.T @ D2) / m
    b2 -= lr * np.mean(D2, axis=0)
    W1 -= lr * (X.T @ D1) / m
    b1 -= lr * np.mean(D1, axis=0)

A1 = relu(X @ W1 + b1)
A2 = relu(A1 @ W2 + b2)
P = sigmoid(A2 @ W3 + b3)

print("Loss:", -np.mean(Y * np.log(P + 1e-10) + (1 - Y) * np.log(1 - P + 1e-10)))
print("Prediction:", (P >= 0.5).astype(int).ravel())
