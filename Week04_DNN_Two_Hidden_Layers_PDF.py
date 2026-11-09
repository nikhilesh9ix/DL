# Week 4 (exam PDF version): Deep Neural Network with Two Hidden Layers (seed 0, stops at tolerance)
import numpy as np

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_loss(y, y_pred):
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

m = int(input("Number of training samples: "))
n = int(input("Number of input features: "))

X = []
y = []
for i in range(m):
    print(f"\nSample {i+1}")
    X.append([float(input(f"X{j+1}: ")) for j in range(n)])
    y.append(int(input("Target (0/1): ")))
X = np.array(X)
y = np.array(y).reshape(m, 1)

hidden1 = int(input("Neurons in hidden layer 1: "))
hidden2 = int(input("Neurons in hidden layer 2: "))
learning_rate = float(input("Learning rate: "))

np.random.seed(0)
W1 = np.random.randn(n, hidden1) * 0.1
B1 = np.zeros((1, hidden1))
W2 = np.random.randn(hidden1, hidden2) * 0.1
B2 = np.zeros((1, hidden2))
W3 = np.random.randn(hidden2, 1) * 0.1
B3 = np.zeros((1, 1))

previous_loss = float("inf")
for iteration in range(1, 10001):
    # Forward propagation
    Z1 = X @ W1 + B1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + B2
    A2 = relu(Z2)
    Z3 = A2 @ W3 + B3
    y_pred = sigmoid(Z3)
    loss = compute_loss(y, y_pred)

    if abs(previous_loss - loss) < 1e-7:
        break
    previous_loss = loss

    # Backpropagation
    dZ3 = y_pred - y
    dW3 = A2.T @ dZ3 / m
    dB3 = np.sum(dZ3, axis=0, keepdims=True) / m

    dZ2 = (dZ3 @ W3.T) * relu_derivative(Z2)
    dW2 = A1.T @ dZ2 / m
    dB2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dZ1 = (dZ2 @ W2.T) * relu_derivative(Z1)
    dW1 = X.T @ dZ1 / m
    dB1 = np.sum(dZ1, axis=0, keepdims=True) / m

    # Update
    W3 -= learning_rate * dW3
    B3 -= learning_rate * dB3
    W2 -= learning_rate * dW2
    B2 -= learning_rate * dB2
    W1 -= learning_rate * dW1
    B1 -= learning_rate * dB1

# Final prediction
A1 = relu(X @ W1 + B1)
A2 = relu(A1 @ W2 + B2)
y_pred = sigmoid(A2 @ W3 + B3)
loss = compute_loss(y, y_pred)
predicted = (y_pred >= 0.5).astype(int)

print("\nTraining completed")
print("Final loss:", round(loss, 8))
print("Iterations:", iteration)
print("\nSample\tActual\tProbability\tPredicted")
for i in range(m):
    print(i + 1, "\t", y[i][0], "\t", round(float(y_pred[i][0]), 6), "\t", predicted[i][0])
