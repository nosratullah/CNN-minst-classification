import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense,Activation
from tensorflow.keras.optimizers import SGD
minst = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = minst.load_data()

x_train[0].shape
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
x_train.shape
28*28
x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)
optimizer = SGD()
model = tf.keras.models.Sequential()

model.add(Dense(10, input_shape=(784,)))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=32, epochs=10, validation_split=0.2)
precisions = model.predict(x_test)

np.argmax(precisions[100])
y_test[100]
