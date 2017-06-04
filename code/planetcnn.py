from __future__ import print_function
import tensorflow.contrib.keras as keras
from tensorflow.contrib.keras import backend as K
import numpy as np

from planetutils import DataHandler

num_classes = 17
imgwidth, imgheight = 256, 256
input_shape = (imgwidth, imgheight, 4)

dh = DataHandler()
dh.set_train_labels()

train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / (np.power(2, 16) - 1),
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / (np.power(2, 16) - 1))

# CNN
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size=(3, 3),
          activation='relu',
          input_shape=input_shape))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(num_classes, activation='sigmoid'))

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

outer_batch_size = 256
inner_batch_size = 128
remix_factor = 4
epochs = 170

for e in range(epochs):
    print('Epoch', e)
    train_iter = dh.get_train_iter()  # One pass through the data == epoch
    x_train = np.empty(shape=(outer_batch_size, imgwidth, imgheight, 4), dtype='float32')
    y_train = np.empty(shape=(outer_batch_size, 17), dtype='bool')
    for i, (X, Y) in enumerate(train_iter):
        x_train[i % outer_batch_size] = X
        y_train[i % outer_batch_size] = Y.drop('name', axis=1).as_matrix()[0]
        if i % outer_batch_size == 0:
            batches = 0
            for x_batch, y_batch in train_datagen.flow(x_train, y_train, batch_size=inner_batch_size):
                print('Batch', batches)
                model.fit(x_batch, y_batch, epochs=4, verbose=1)
                batches += 1
                if batches >= outer_batch_size // inner_batch_size * remix_factor:
                    break
