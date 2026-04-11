import tensorflow as tf
import numpy as np


(x_train , y_train) , (x_test , y_test) = tf.keras.datasets.imdb.load_data(num_words = 10000)
print(x_train.shape)
print(x_train)


from tensorflow.keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(x_train , maxlen=200)
x_test = pad_sequences(x_test , maxlen=200)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM , Embedding , Dense , Dropout


model = Sequential([
    Embedding(10000 ,32),
    LSTM(64),
    Dense(128,activation = "relu"),
    Dropout(0.5),
    Dense(1 , activation = "sigmoid")]
)


from tensorflow.keras.callbacks import EarlyStopping
Early_stop = EarlyStopping(monitor = "val_accuracy" , patience = 3 ,restore_best_weights = True)


model.compile(
    optimizer = "adam",
    loss = "binary_crossentropy",
    metrics = ["accuracy"]
)


model.fit(x_train,y_train , epochs =5 , validation_data = (x_test,y_test) , callbacks = [Early_stop])


import re
word_index = tf.keras.datasets.imdb.get_word_index()


def predict_sentiment(review):
    review = review.lower()
    review = re.sub(r"[^a-z\s]","",review)
    review = review.split()
    sequence = [word_index.get(word,2) for word in review]
    print("sequence:", sequence[:10])
    padded = pad_sequences([sequence] , maxlen=200)
    prediction = model.predict(padded)[0][0]
    if prediction > 0.5:
        print(f"Positive with confidence: {prediction:.0%}")
    else:
        print(f"Negative with confidence: {1-prediction:.0%}")
predict_sentiment("Formula one was a very weird movie. On one hand, it was very confusing, but it was also compelling")
model.save("sentiment_analyzer.keras")
import json
word_index = tf.keras.datasets.imdb.get_word_index()
with open("word_index.json","w") as f:
    json.dump(word_index , f)
print("model and index saved!")
predict_sentiment("i didn't enjoy the movie at all. it was very boring and lame.")
predict_sentiment("bad")
predict_sentiment("good")
predict_sentiment("this was the worst most terrible awful disgusting horrible movie ever made in the history of cinema and i hated every single second of it")
predict_sentiment("great")