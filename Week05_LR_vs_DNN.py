# Week 5: Compare Logistic Regression and Deep Neural Network
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras import Sequential, layers

n = int(input("Samples: "))
noise = float(input("Noise: "))

X, Y = make_moons(n_samples=n, noise=noise, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# Deep Neural Network
model = Sequential([
    layers.Input(shape=(2,)),
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(8, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=100, verbose=0)
dnn_pred = (model.predict(X_test, verbose=0) >= 0.5).astype(int).ravel()

for name, p in [("Logistic Regression", lr_pred), ("DNN", dnn_pred)]:
    print("\n", name)
    print("Accuracy:", accuracy_score(y_test, p))
    print("Precision:", precision_score(y_test, p, zero_division=0))
    print("Recall:", recall_score(y_test, p, zero_division=0))
    print("F1 Score:", f1_score(y_test, p, zero_division=0))
