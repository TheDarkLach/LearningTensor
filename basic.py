import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib


data = keras.datasets.fashion_mnist

# split data into training and testing data.
(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# divide the data by 255 to get a value between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# build the model, 784 input neurons(one for each pixel), 128 hidden neurons, 10 output neurons (one for each class)
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

#training the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

#5 epochs 0.871399998664856

model.fit(train_images, train_labels, epochs=10)

"""test_loss, test_acc = model.evaluate(test_images, test_labels)

print('\nTest accuracy:', test_acc)"""

predictions = model.predict(test_images)

plt.figure(figsize=(5,5))
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediction: " + class_names[np.argmax(predictions[i])])
    plt.show()