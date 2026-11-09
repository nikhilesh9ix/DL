# Basics: Perceptron Learning Algorithm
from itertools import product

def activation(z):
    return 1 if z >= 0 else 0

n = int(input("Number of inputs: "))
weights = [float(input(f"W{i+1}: ")) for i in range(n)]
bias = float(input("Bias: "))
learning_rate = float(input("Learning rate: "))

dataset = list(product([0, 1], repeat=n))
targets = [int(input(f"Target for {row}: ")) for row in dataset]

converged = False
for epoch in range(1, 101):
    errors = 0
    for row, target in zip(dataset, targets):
        z = sum(row[i] * weights[i] for i in range(n)) + bias
        prediction = activation(z)
        error = target - prediction
        if error != 0:
            errors += 1
            for i in range(n):
                weights[i] += learning_rate * error * row[i]
            bias += learning_rate * error
    if errors == 0:
        converged = True
        break

print("\nInputs\tTarget\tOutput")
for row, target in zip(dataset, targets):
    z = sum(row[i] * weights[i] for i in range(n)) + bias
    print(row, "\t", target, "\t", activation(z))

print("\nFinal weights:", [round(w, 3) for w in weights])
print("Final bias:", round(bias, 3))
print("Epochs:", epoch)
if converged:
    print("Status: Converged")
else:
    print("Status: Not linearly separable")
