# Week 1: Single Neuron Perceptron with Normalization and Activation Functions
import math

n = int(input("Number of inputs: "))
x = [float(input(f"X{i+1}: ")) for i in range(n)]
w = [float(input(f"W{i+1}: ")) for i in range(n)]
b = float(input("Bias: "))

z = sum(x[i] * w[i] for i in range(n)) + b

print("0.None 1.MinMax01 2.MinMax(-1,1) 3.Zscore 4.Decimal")
c = int(input("Normalization: "))

xmin, xmax = min(x), max(x)
xn = x

if c == 1:
    xn = [(v - xmin) / (xmax - xmin) for v in x]
elif c == 2:
    xn = [2 * (v - xmin) / (xmax - xmin) - 1 for v in x]
elif c == 3:
    mean = sum(x) / n
    sd = (sum((v - mean) ** 2 for v in x) / n) ** 0.5
    xn = [(v - mean) / sd for v in x]
elif c == 4:
    biggest = max(abs(v) for v in x)
    j = len(str(int(biggest)))
    xn = [v / 10 ** j for v in x]

zn = sum(xn[i] * w[i] for i in range(n)) + b

print("1.Linear 2.Step 3.Sigmoid 4.Tanh 5.ReLU")
print("6.Leaky ReLU 7.ELU 8.Softplus")
a = int(input("Activation: "))

def activate(z):
    if a == 1: return z
    if a == 2: return 1 if z >= 0 else 0
    if a == 3: return 1 / (1 + math.exp(-z))
    if a == 4: return math.tanh(z)
    if a == 5: return max(0, z)
    if a == 6: return z if z >= 0 else 0.01 * z
    if a == 7: return z if z >= 0 else math.exp(z) - 1
    return math.log(1 + math.exp(z))

print("Without normalization:", activate(z))
print("With normalization:", activate(zn))
