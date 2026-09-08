import numpy as np
from scipy.special import expit, softmax
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# STEP 1 - sample raw data (one row, 4 values)
x = np.array([[2.0, 8.0, -3.0, 5.0]])

# STEP 2 - normalization
minmax = MinMaxScaler().fit_transform(x.T).T
zscore = StandardScaler().fit_transform(x.T).T

print("Original :", x[0])
print("Min-Max  :", minmax[0])
print("Z-Score  :", zscore[0])

# STEP 3 - basic activation functions
z = x[0]
sig = expit(z)  # sigmoid

print("\nSigmoid  :", sig)
print("dSigmoid :", sig * (1 - sig))  # derivative of sigmoid
print("Softmax  :", softmax(z))
print("Tanh     :", np.tanh(z))
print("ReLU     :", np.maximum(0, z))
