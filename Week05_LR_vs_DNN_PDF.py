# Week 5 (exam PDF version): DNN vs Logistic Regression on Two-Moons (1000 samples, stratified split)
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras import Sequential, layers

X, y = make_moons(n_samples=1000, noise=0.20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def show_results(name, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    print("\n" + name)
    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision_score(y_test, y_pred) * 100, 2), "%")
    print("Recall:", round(recall_score(y_test, y_pred) * 100, 2), "%")
    print("F1 Score:", round(f1_score(y_test, y_pred) * 100, 2), "%")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    return accuracy

# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# Deep Neural Network
dnn_model = Sequential([
    layers.Input(shape=(2,)),
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(8, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
dnn_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
dnn_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.20, verbose=0)
probability = dnn_model.predict(X_test, verbose=0)
dnn_pred = (probability >= 0.5).astype(int).flatten()

lr_accuracy = show_results("Logistic Regression", lr_pred)
dnn_accuracy = show_results("Deep Neural Network", dnn_pred)

if dnn_accuracy > lr_accuracy:
    print("\nDNN performed better than Logistic Regression")
elif dnn_accuracy < lr_accuracy:
    print("\nLogistic Regression performed better than DNN")
else:
    print("\nBoth models achieved the same accuracy")
