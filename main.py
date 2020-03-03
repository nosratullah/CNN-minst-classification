import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Downloading MINST dataset directly from tensorflow library
minst = tf.keras.datasets.mnist
# Assigning data to arrays
(x_train, y_train),(x_test, y_test) = minst.load_data()
# Normalizing arrays
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
# Create an empty sequential model
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())
# Creating the first, second and third layer with 128, 128 and 19 neuron using relu, relu and softmax activation function
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
# Compiling the model with adam optimizer
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',metrics=['accuracy'])
# Start the training with 3 itteration
model.fit(x_train, y_train,epochs=3)

val_loss, val_acc = model.evaluate(x_test, y_test)
print(round(val_loss,4), val_acc)
# Save the model for later uses
model.save('digit_reader.model')

# Testing model
new_model = tf.keras.models.load_model('digit_reader.model')
predictions = new_model.predict(x_test)



#np.argmax(predictions[0])
#len(x_test)
# Here I used from test dataset to test out whether the model can predict or not!
plt.figure(figsize=(15,10))
for i in range(10):
    random = np.random.randint(0,1000)
    plt.subplot(5,2,i+1)
    plt.imshow(x_test[random])
    plt.ylabel(np.argmax(predictions[random]))
plt.savefig('mints_dataset.png')
