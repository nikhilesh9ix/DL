# Week 8: CNN on MNIST with Convolution and Pooling Demonstrations
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.datasets import mnist

# Pooling demo: numbers 1 to 16 as a 4x4 image
feature = np.arange(1, 17, dtype=np.float32).reshape(1, 4, 4, 1)
print("Max Pooling:")
print(tf.nn.max_pool2d(feature, 2, 2, "VALID").numpy()[0, :, :, 0])
print("Average Pooling:")
print(tf.nn.avg_pool2d(feature, 2, 2, "VALID").numpy()[0, :, :, 0])

# Convolution demo: 7x7 black image with a white block
image = np.zeros((1, 7, 7, 1), dtype=np.float32)
image[0, 1:5, 2:5, 0] = 1

kernels = {
    "Vertical Edge": [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
    "Horizontal Edge": [[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
    "Sharpening": [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
}

plt.figure(figsize=(12, 3))
plt.subplot(1, 4, 1)
plt.imshow(image[0, :, :, 0], cmap="gray")
plt.title("Original Image")
plt.axis("off")
for i, name in enumerate(kernels):
    kernel = np.array(kernels[name], dtype=np.float32).reshape(3, 3, 1, 1)
    output = tf.nn.conv2d(image, kernel, strides=1, padding="SAME")
    plt.subplot(1, 4, i + 2)
    plt.imshow(output[0, :, :, 0], cmap="gray")
    plt.title(name)
    plt.axis("off")
plt.tight_layout()
plt.show()

# MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

# CNN model
model = Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(2),
    layers.Conv2D(64, 3, activation="relu", padding="same"),
    layers.AveragePooling2D(2),
    layers.Conv2D(128, 3, activation="relu", padding="same"),
    layers.GlobalAveragePooling2D(),
    layers.Dense(10, activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1)
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print("Test Loss:", round(loss, 4))
print("Test Accuracy:", round(acc * 100, 2), "%")

plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("CNN Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
