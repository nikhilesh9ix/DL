import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import SGD

# STEP 1 - dataset
x, y = make_classification(n_samples=200, n_features=2, n_redundant=0, random_state=42)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

# STEP 2 - logistic regression AS a single-neuron neural network
model = Sequential()
model.add(Input(shape=(2,)))
model.add(Dense(units=1, activation="sigmoid"))  # 1 neuron + sigmoid = logistic regression

# STEP 3 - compile: binary_crossentropy = cost function, SGD = gradient descent update rule
model.compile(optimizer=SGD(learning_rate=0.1), loss="binary_crossentropy", metrics=["accuracy"])

# STEP 4 - train (minimizes cost, updates weights+bias every epoch)
history = model.fit(xtrain, ytrain, epochs=100, verbose=0)

# STEP 5 - results
loss, acc = model.evaluate(xtest, ytest, verbose=0)
print("Test Accuracy :", acc)
print("Final Weights :", model.get_weights()[0].flatten())
print("Final Bias    :", model.get_weights()[1])
print("Cost (first 5 epochs):", [round(c, 4) for c in history.history["loss"][:5]])
print("Cost (last 5 epochs) :", [round(c, 4) for c in history.history["loss"][-5:]])
