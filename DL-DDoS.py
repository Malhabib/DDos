import os

import arff
import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import  MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler

file = open("/content/drive/My Drive/data/final-dataset.arff", 'r')


def cnn_model(shape, FILTERS):
    model = Sequential()
    model.add(Conv2D(filters=FILTERS, kernel_size=2, activation='relu', padding='same', input_shape=shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=1))
    model.add(Conv2D(filters=FILTERS, kernel_size=2, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=1))
    model.add(Conv2D(filters=FILTERS, kernel_size=2, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=1))
    model.add(Conv2D(filters=FILTERS, kernel_size=2, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=1))
    model.add(Conv2D(filters=FILTERS, kernel_size=2, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=1))
    model.add(Conv2D(filters=FILTERS, kernel_size=2, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=1))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=1))
    model.add(Flatten())
    # model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(Dense(5, activation='softmax'))

    return model


def scrape_data():
    # decode the .arff data and change text labels into numerical
    decoder = arff.ArffDecoder()
    data = decoder.decode(file, encode_nominal=True)

    # split the raw data into data and labels
    vals = [val[0: -1] for val in data['data']]
    labels = [label[-1] for label in data['data']]

    for val in labels:
        if labels[val] != 0:
            labels[val] = 1

    # split the labels and data into traning and validation sets
    training_data = vals[0: int(.9 * len(vals))]
    training_labels = labels[0: int(.9 * len(vals))]
    validation_data = vals[int(.9 * len(vals)):]
    validation_labels = labels[int(.9 * len(vals)):]

    print(training_labels)

    # flatten labels with one hot encoding
    training_labels = to_categorical(training_labels, 5)
    validation_labels = to_categorical(validation_labels, 5)

    # save all arrays with numpy
    np.save('/content/drive/My Drive/data/saved-files/vals', np.asarray(vals))
    np.save('/content/drive/My Drive/data/saved-files/labels', np.asarray(labels))
    np.save('/content/drive/My Drive/data/saved-files/training_data', np.asarray(training_data))
    np.save('/content/drive/My Drive/data/saved-files/validation_data', np.asarray(validation_data))
    np.save('/content/drive/My Drive/data/saved-files/training_labels', np.asarray(training_labels))
    np.save('/content/drive/My Drive/data/saved-files/validation_labels', np.asarray(validation_labels))


# check to see if saved data exists, if not then create the data
if not os.path.exists('/content/drive/My Drive/data/saved-files/training_data.npy') or not os.path.exists(
        '/content/drive/My Drive/data/saved-files/training_labels.npy') or not os.path.exists(
    '/content/drive/My Drive/data/saved-files/validation_data.npy') or not os.path.exists(
    '/content/drive/My Drive/data/saved-files/validation_labels.npy'):
    print('creating')
    if not os.path.exists('/content/drive/My Drive/data/saved-files'):
        os.mkdir('/content/drive/My Drive/data/saved-files')
    scrape_data()

# load the saved data
data_train = np.load('/content/drive/My Drive/data/saved-files/training_data.npy')
label_train = np.load('/content/drive/My Drive/data/saved-files/training_labels.npy')
data_eval = np.load('/content/drive/My Drive/data/saved-files/validation_data.npy')
label_eval = np.load('/content/drive/My Drive/data/saved-files/validation_labels.npy')

# scaling the data
scaler = MinMaxScaler()
scaler.fit(data_train)
data_train = scaler.transform(data_train)
data_eval = scaler.transform(data_eval)

# reshaping data
data_train = data_train.reshape(data_train.shape[0], 3, 9, 1)
data_eval = data_eval.reshape(data_eval.shape[0], 3, 9, 1)

FILTERS = 32
# generate and compile the model
model = cnn_model(data_train.shape[1:4], FILTERS)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# initialize tensorboard
tensorboard = TensorBoard(log_dir='logs', histogram_freq=0, write_graph=True, write_images=True)

# training starts here
history = model.fit(data_train, label_train, validation_data=(data_eval, label_eval), epochs=100,
                    callbacks=[tensorboard])
loss_history = history.history["loss"]

numpy_loss_history = np.array(loss_history)
np.savetxt("/content/drive/My Drive/data/saved-files/loss_history.txt", numpy_loss_history, delimiter=",")

# evaluating the model's performace
print('overall testing accuracy: ', model.evaluate(data_eval, label_eval)[1])
print('overall training accuracy: ', model.evaluate(data_train, label_train)[1])

# if create_model_image:
plot_model(model, to_file='model.png', show_shapes=True)

plt.figure(1)

# summarize history for accuracy
plt.subplot(211)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# play sound when done with code to alert me
os.system('afplay /System/Library/Sounds/Ping.aiff')
os.system('afplay /System/Library/Sounds/Ping.aiff')
