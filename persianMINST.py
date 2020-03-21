import numpy as np
from HodaDatasetReader import read_hoda_cdb, read_hoda_dataset
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten

train_images, train_labels = read_hoda_dataset('./DigitDB/Train 60000.cdb', reshape=False)
test_images, test_labels = read_hoda_dataset('./DigitDB/Test 20000.cdb', reshape=False)
remaining_images, remaining_labels = read_hoda_dataset('./DigitDB/RemainingSamples.cdb', reshape=False)

model = tf.keras.models.Sequential()

model.add(Conv2D(64, (3, 3), input_shape = train_images.shape[1:]))
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels,epochs=3)

model.save('persianMINST.model')
score = model.evaluate(test_images, test_labels)
