import numpy as np

def tanh(x):
    return np.tanh(x)

def sigmoid(x):
    return 1/(1+np.exp(-x))

m=int(input("Samples: "))
n=int(input("Features: "))
h=int(input("Hidden neurons: "))

X=np.array([[float(input(f"X{j+1}: ")) for j in range(n)] for i in range(m)])
Y=np.array([int(input("Target: ")) for i in range(m)]).reshape(-1,1)

W1=np.array([float(input(f"W1_{i+1}: ")) for i in range(n*h)]).reshape(n,h)
W2=np.array([float(input(f"W2_{i+1}: ")) for i in range(h)]).reshape(h,1)

b1=float(input("Bias 1: "))
b2=float(input("Bias 2: "))
lr=float(input("Learning rate: "))

for i in range(10000):

    # Forward
    A1=tanh(np.dot(X,W1)+b1)
    P=sigmoid(np.dot(A1,W2)+b2)

    # Backpropagation
    D2=P-Y
    D1=np.dot(D2,W2.T)*(1-A1**2)

    # Update
    W2-=lr*np.dot(A1.T,D2)/m
    b2-=lr*np.mean(D2)
    W1-=lr*np.dot(X.T,D1)/m
    b1-=lr*np.mean(D1)

P=sigmoid(np.dot(tanh(np.dot(X,W1)+b1),W2)+b2)

loss=-np.mean(Y*np.log(P+1e-10)+(1-Y)*np.log(1-P+1e-10))

print("Loss:",loss)
print("Prediction:",(P>=.5).astype(int).ravel())