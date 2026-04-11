import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.python.keras.saving.saved_model.load import input_layer

mnist = tf.keras.datasets.mnist
(x_train , y_train) , (x_test , y_test) = mnist.load_data()


print(x_train.shape)
print(y_train.shape)
print(x_train[0] , y_train[0])


for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(x_train[i] , cmap ="gray")
    plt.title(str(y_train[i]))
    plt.axis("off")
plt.show()


x_train = x_train / 255.0
x_test = x_test / 255.0


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = (28,28)),
    tf.keras.layers.Dense(128 , activation = "relu"),
    tf.keras.layers.Dense(128 , activation = "relu"),
    tf.keras.layers.Dense(64 , activation = "relu"),
    tf.keras.layers.Dense(10 , activation = "softmax")
])


model.compile(
    optimizer = "adam",
    loss = "sparse_categorical_crossentropy",
    metrics = ["accuracy"]
)


model.fit(x_train , y_train , epochs = 5 , validation_data = (x_test , y_test))

predictions = model.predict(x_test)
for i in range(7):
    plt.subplot(1,7,i+1)
    plt.imshow(x_test[i] , cmap ="gray")
    plt.axis("off")
    plt.title(f"pred: {np.argmax(predictions[i])}")
plt.show()