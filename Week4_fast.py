import numpy as np

def relu(x):
    return np.maximum(0,x)

def sigmoid(x):
    return 1/(1+np.exp(-x))

m=int(input("Samples: "))
n=int(input("Features: "))

X=np.array([[float(input(f"X{j+1}: ")) for j in range(n)] for i in range(m)])
Y=np.array([int(input("Target: ")) for i in range(m)]).reshape(-1,1)

h1=int(input("Hidden layer 1: "))
h2=int(input("Hidden layer 2: "))
lr=float(input("Learning rate: "))

W1=np.random.randn(n,h1)*.1
W2=np.random.randn(h1,h2)*.1
W3=np.random.randn(h2,1)*.1
b1=np.zeros((1,h1))
b2=np.zeros((1,h2))
b3=np.zeros((1,1))

for i in range(10000):

    # Forward
    A1=relu(np.dot(X,W1)+b1)
    A2=relu(np.dot(A1,W2)+b2)
    P=sigmoid(np.dot(A2,W3)+b3)

    # Backpropagation
    D3=P-Y
    D2=np.dot(D3,W3.T)*(A2>0)
    D1=np.dot(D2,W2.T)*(A1>0)

    # Update
    W3-=lr*np.dot(A2.T,D3)/m
    b3-=lr*np.mean(D3,axis=0)
    W2-=lr*np.dot(A1.T,D2)/m
    b2-=lr*np.mean(D2,axis=0)
    W1-=lr*np.dot(X.T,D1)/m
    b1-=lr*np.mean(D1,axis=0)

P=sigmoid(np.dot(relu(np.dot(relu(np.dot(X,W1)+b1),W2)+b2),W3)+b3)

print("Loss:",-np.mean(Y*np.log(P+1e-10)+(1-Y)*np.log(1-P+1e-10)))
print("Prediction:",(P>=.5).astype(int).ravel())