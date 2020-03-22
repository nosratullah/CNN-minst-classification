import numpy as np
from HodaDatasetReader import read_hoda_cdb, read_hoda_dataset
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten
from sklearn.utils import shuffle
import pickle

train_images, train_labels = read_hoda_dataset('./DigitDB/Train 60000.cdb', reshape=False)
test_images, test_labels = read_hoda_dataset('./DigitDB/Test 20000.cdb', reshape=False)
remaining_images, remaining_labels = read_hoda_dataset('./DigitDB/RemainingSamples.cdb', reshape=False)
# because of the dataset, it's better to shuffle the dataset to increase accuracy
train_images, train_labels = shuffle(np.array(train_images), np.array(train_labels))
test_images, test_labels = shuffle(np.array(test_images), np.array(test_labels))
remaining_images, remaining_labels = shuffle(np.array(remaining_images), np.array(remaining_labels))

# In order to save dataset with pickle
'''
listNames = ['train_images', 'train_labels', 'test_images', 'test_labels', 'remaining_images', 'remaining_labels']
for i in listNames:
    pickle_out = open("DigitDB/{}.pickle".format(i), 'wb')
    pickle.dump(i, pickle_out)
    pickle_out.close()

# Load the dataset after saving it
pickle_in = open("DigitDB/train_images.pickle", "rb")
train_images = pickle.load(pickle_in)
'''
#plt.imshow(np.reshape(test_images[903], (32, 32)))
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

prediction = model.predict(test_images)

plt.figure(figsize=(15,8))
for i in range(1,11):
    r = np.random.randint(0,len(test_labels))
    plt.subplot(2,5,i)
    plt.imshow(np.reshape(test_images[r], (32, 32)))
    plt.title(np.argmax(prediction[r]))

plt.savefig('persianMINST.png',dpi=200)
