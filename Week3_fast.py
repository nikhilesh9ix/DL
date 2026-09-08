import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# STEP 1 - two-class dataset
x, y = make_moons(n_samples=300, noise=0.2, random_state=42)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.transform(xtest)

# STEP 2 - single hidden layer, tanh (non-linear), sigmoid output (two-class)
model = Sequential()
model.add(Input(shape=(2,)))
model.add(Dense(units=6, activation="tanh"))   # hidden layer - tanh
model.add(Dense(units=1, activation="sigmoid"))  # output layer - two-class probability

# STEP 3 - cross-entropy loss, backprop done via Keras optimizer
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# STEP 4 - train
history = model.fit(xtrain, ytrain, epochs=60, verbose=0)

# STEP 5 - results
loss, acc = model.evaluate(xtest, ytest, verbose=0)
print("Test Accuracy :", acc)
print("Cross-Entropy Loss (final):", loss)
print("Loss curve (first 5):", [round(c, 4) for c in history.history["loss"][:5]])
print("Loss curve (last 5) :", [round(c, 4) for c in history.history["loss"][-5:]])
