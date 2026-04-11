import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , MaxPooling2D , GlobalAveragePooling2D , Input


(x_train , y_train) , (x_test,y_test) = tf.keras.datasets.cifar10.load_data()
x_train = tf.image.resize(x_train,(96,96))/255.0
x_test = tf.image.resize(x_test,(96,96))/255.0


labels = ["airplane", "car", "bird", "cat", "deer",
          "dog", "frog", "horse", "ship", "truck"]


base_model = MobileNetV2(include_top = False, weights = "imagenet" , input_shape=(96,96,3))
base_model.trainable = False


model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128 , activation="relu"),
    Dense(10,activation = "softmax")
])


model.summary()


from tensorflow.keras.callbacks import EarlyStopping
early=EarlyStopping(monitor = "val_accuracy" , patience = 3 ,restore_best_weights = True)


model.compile(optimizer = "adam" ,
              loss = "sparse_categorical_crossentropy",
              metrics = ["accuracy"])


model.fit(x_train,y_train , validation_data = (x_test,y_test) , epochs = 5 , callbacks = [early])