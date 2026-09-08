from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

n = int(input("Samples: "))
noise = float(input("Noise: "))

X,Y = make_moons(n_samples=n,noise=noise,random_state=42)

X1,X2,Y1,Y2 = train_test_split(X,Y,test_size=.2,random_state=42)

s = StandardScaler()
X1 = s.fit_transform(X1)
X2 = s.transform(X2)

# Logistic Regression
lr = LogisticRegression()
lr.fit(X1,Y1)
p1 = lr.predict(X2)

# Deep Neural Network
dnn = Sequential([
    Dense(32,activation="relu",input_shape=(2,)),
    Dense(16,activation="relu"),
    Dense(8,activation="relu"),
    Dense(1,activation="sigmoid")
])

dnn.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
dnn.fit(X1,Y1,epochs=100,verbose=0)

p2 = (dnn.predict(X2,verbose=0) >= .5).astype(int).ravel()

print("Logistic Regression Accuracy:",accuracy_score(Y2,p1))
print("DNN Accuracy:",accuracy_score(Y2,p2))