from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

#import & load fashion MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#setting category names to map to 0-9 classification values
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#scale image values to 0-1
train_images = train_images / 255.0
test_images = test_images / 255.0

#display first 25 images in dataset with labels.
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

#training model on training data.
history = model.fit(train_images, train_labels, epochs=5)

#evaluating trained model
print("Evaluating trained model...\n")
test_loss, test_acc = model.evaluate(test_images, test_labels)