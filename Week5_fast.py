import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# STEP 1 - any classification dataset
x, y = make_classification(n_samples=500, n_features=10, random_state=42)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.transform(xtest)

# STEP 2 - Logistic Regression
lr = LogisticRegression().fit(xtrain, ytrain)
lr_acc = accuracy_score(ytest, lr.predict(xtest))

# STEP 3 - Deep Neural Network
model = Sequential()
model.add(Input(shape=(10,)))
model.add(Dense(units=16, activation="relu"))
model.add(Dense(units=8, activation="relu"))
model.add(Dense(units=1, activation="sigmoid"))
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(xtrain, ytrain, epochs=50, verbose=0)

dnn_pred = (model.predict(xtest, verbose=0) >= 0.5).astype(int).flatten()
dnn_acc = accuracy_score(ytest, dnn_pred)

# STEP 4 - compare
print("Logistic Regression Accuracy :", lr_acc)
print("Deep Neural Network Accuracy :", dnn_acc)
print("Better Model:", "DNN" if dnn_acc > lr_acc else "Logistic Regression" if lr_acc > dnn_acc else "Tie")
