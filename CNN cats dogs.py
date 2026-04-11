import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D , MaxPooling2D , Flatten , Dense , Input , BatchNormalization , Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
(x_train , y_train) , (x_test , y_test) = tf.keras.datasets.cifar10.load_data()
labels = ["airplane", "car", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(x_train[i])
    plt.axis("off")
    plt.title(labels[y_train[i][0]])
plt.show()
x_train = x_train/255.0
x_test = x_test/255.0
EarlyStopping(monitor ="val_accuracy" , patience = 3 , restore_best_weights = True)
model = Sequential([
    Input(shape=(32,32,3)),
    Conv2D(32,(3,3) ,activation = "relu"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(64,(3,3),activation = "relu"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),
    Dense(64,activation = "relu"),
    Dropout(0.5),
    Dense(10,activation = "softmax")

])
print(model.summary())
model.compile(optimizer = "adam" , loss = "sparse_categorical_crossentropy" , metrics = ["accuracy"])
model.fit(x_train , y_train , epochs = 12 , validation_data = (x_test , y_test))
predictions = model.predict(x_test)
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(x_test[i])
    plt.axis("off")
    plt.title(f"{labels[np.argmax(predictions[i])]} \n actual: {y_test[i][0]}")
plt.show()