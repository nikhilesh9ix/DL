# Week 1 (exam PDF version): Normalization and Activation Functions
import math

n = int(input("Number of inputs: "))
x = [float(input(f"X{i+1}: ")) for i in range(n)]
weights = [float(input(f"W{i+1}: ")) for i in range(n)]
bias = float(input("Bias: "))

print("\n0. No Normalization")
print("1. Min-Max (0 to 1)")
print("2. Min-Max (-1 to 1)")
print("3. Z-Score")
print("4. Decimal Scaling")
norm = int(input("Choose normalization (0-4): "))

x_norm = x
norm_name = "No Normalization"

if norm == 1 or norm == 2:
    xmin = float(input("Minimum value: "))
    xmax = float(input("Maximum value: "))

if norm == 1:
    x_norm = [(v - xmin) / (xmax - xmin) for v in x]
    norm_name = "Min-Max (0 to 1)"
elif norm == 2:
    x_norm = [2 * (v - xmin) / (xmax - xmin) - 1 for v in x]
    norm_name = "Min-Max (-1 to 1)"
elif norm == 3:
    mean = sum(x) / n
    std = (sum((v - mean) ** 2 for v in x) / n) ** 0.5
    x_norm = [(v - mean) / std for v in x]
    norm_name = "Z-Score"
elif norm == 4:
    j = int(input("Decimal scaling factor j: "))
    x_norm = [v / (10 ** j) for v in x]
    norm_name = "Decimal Scaling"

print("\n1. Linear")
print("2. Binary Step")
print("3. Sigmoid")
print("4. Tanh")
print("5. ReLU")
print("6. Leaky ReLU")
print("7. ELU")
print("8. Softplus")
choice = int(input("Choose activation (1-8): "))

def activation(z, choice):
    if choice == 2:
        return 1 if z >= 0 else 0
    elif choice == 3:
        return 1 / (1 + math.exp(-z))
    elif choice == 4:
        return math.tanh(z)
    elif choice == 5:
        return max(0, z)
    elif choice == 6:
        return z if z > 0 else 0.01 * z
    elif choice == 7:
        return z if z > 0 else math.exp(z) - 1
    elif choice == 8:
        return math.log(1 + math.exp(z))
    else:
        return z

z = sum(x[i] * weights[i] for i in range(n)) + bias
z_norm = sum(x_norm[i] * weights[i] for i in range(n)) + bias

a = activation(z, choice)
a_norm = activation(z_norm, choice)

print("\nOriginal inputs:", [round(v, 6) for v in x])
print("Normalized inputs:", [round(v, 6) for v in x_norm])
print("Normalization:", norm_name)
print("\nWithout normalization:")
print("Weighted sum:", round(z, 6))
print("Activation output:", round(a, 6))
print("\nWith normalization:")
print("Weighted sum:", round(z_norm, 6))
print("Activation output:", round(a_norm, 6))
