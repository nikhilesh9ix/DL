# Basics: McCulloch-Pitts Neuron Model
from itertools import product

n = int(input("Number of inputs: "))
weights = [float(input(f"W{i+1}: ")) for i in range(n)]
threshold = float(input("Threshold: "))

dataset = list(product([0, 1], repeat=n))

print("\nInputs\tWeighted Sum\tOutput")
for inputs in dataset:
    weighted_sum = sum(inputs[i] * weights[i] for i in range(n))
    output = 1 if weighted_sum >= threshold else 0
    print(inputs, "\t", round(weighted_sum, 2), "\t\t", output)
