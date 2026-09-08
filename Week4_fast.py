import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# STEP 1 - classification dataset
x, y = make_classification(n_samples=400, n_features=10, n_informative=6, random_state=42)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.transform(xtest)

# STEP 2 - DEEP network: more than one hidden layer, ReLU (non-linear)
model = Sequential()
model.add(Input(shape=(10,)))
model.add(Dense(units=16, activation="relu"))  # hidden layer 1 - ReLU
model.add(Dense(units=8, activation="relu"))   # hidden layer 2 - ReLU
model.add(Dense(units=1, activation="sigmoid"))  # output layer

# STEP 3 - compile + train
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
history = model.fit(xtrain, ytrain, epochs=50, verbose=0)

# STEP 4 - results
loss, acc = model.evaluate(xtest, ytest, verbose=0)
print("Test Accuracy :", acc)
print("Final Loss    :", loss)
print("Loss curve (first 5):", [round(c, 4) for c in history.history["loss"][:5]])
print("Loss curve (last 5) :", [round(c, 4) for c in history.history["loss"][-5:]])
model.summary()
