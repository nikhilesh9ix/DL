import math

n = int(input("Enter number of inputs: "))
x = [float(input(f"X{i+1}: ")) for i in range(n)]
w = [float(input(f"W{i+1}: ")) for i in range(n)]
b = float(input("Enter bias: "))

print("\n1.Min-Max 0-1  2.Min-Max -1-1  3.Z-Score  4.Decimal  0.None")
c = int(input("Normalization choice: "))

if c == 1:
    mn = float(input("Min: "))
    mx = float(input("Max: "))
    xn = [(i-mn)/(mx-mn) for i in x]
elif c == 2:
    mn = float(input("Min: "))
    mx = float(input("Max: "))
    xn = [2*(i-mn)/(mx-mn)-1 for i in x]
elif c == 3:
    m = sum(x)/n
    s = (sum((i-m)**2 for i in x)/n)**0.5
    xn = [(i-m)/s if s else 0 for i in x]
elif c == 4:
    j = int(input("Enter scaling factor: "))
    xn = [i/(10**j) for i in x]
else:
    xn = x.copy()

print("\n1.Linear 2.Step 3.Sigmoid 4.Tanh 5.ReLU 6.Leaky 7.ELU 8.Softplus")
a = int(input("Activation choice: "))

def f(z):
    if a == 1: return z
    if a == 2: return 1 if z >= 0 else 0
    if a == 3: return 1/(1+math.exp(-z))
    if a == 4: return math.tanh(z)
    if a == 5: return max(0,z)
    if a == 6: return z if z > 0 else .01*z
    if a == 7: return z if z > 0 else math.exp(z)-1
    return math.log(1+math.exp(z))

z = sum(x[i]*w[i] for i in range(n)) + b
zn = sum(xn[i]*w[i] for i in range(n)) + b

print("\nOriginal:", x)
print("Normalized:", xn)
print("Without Normalization: Z =", z, " Output =", f(z))
print("With Normalization: Z =", zn, " Output =", f(zn))