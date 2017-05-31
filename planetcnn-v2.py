from __future__ import print_function
import tensorflow.contrib.keras as keras
from tensorflow.contrib.keras import backend as K
import numpy as np

from planetutils import DataHandler

atmos = ['cloudy', 'partly_cloudy', 'haze', 'clear']
landuse = ['artisinal_mine', 'blooming', 'blow_down', 'agriculture', 'bare_ground',
           'primary', 'road', 'selective_logging', 'slash_burn', 'water',
           'conventional_mine', 'cultivation', 'habitation']

num_classes_atmos = len(atmos)
num_classes_landuse = len(landuse)
imgwidth, imgheight, numchans = 256, 256, 4
input_shape = (imgwidth, imgheight, numchans)

dh = DataHandler()
dh.set_train_labels()

# Setup image data generators
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / (np.power(2, 16) - 1),
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / (np.power(2, 16) - 1))

# Setup tensorboard callbacks
graphdir = "/graph"
tb_cb_model_atmos = keras.callbacks.TensorBoard(log_dir=dh.basepath + graphdir + "/atmos", histogram_freq=0,
                                                write_graph=True, write_images=True)
tb_cb_model_haze = keras.callbacks.TensorBoard(log_dir=dh.basepath + graphdir + "/haze", histogram_freq=0,
                                               write_graph=True, write_images=True)
tb_cb_model_partly_cloudy = keras.callbacks.TensorBoard(log_dir=dh.basepath + graphdir + "/partly-cloudy", histogram_freq=0,
                                                        write_graph=True, write_images=True)
tb_cb_model_clear = keras.callbacks.TensorBoard(log_dir=dh.basepath + graphdir + "/clear", histogram_freq=0,
                                                write_graph=True, write_images=True)

# Setup model checkpoint callbacks
modeldir = "/model/"
mc_cb_model_atmos = keras.callbacks.ModelCheckpoint(dh.basepath + modeldir + 'atmos.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss',
                                                    verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
mc_cb_model_haze = keras.callbacks.ModelCheckpoint(dh.basepath + modeldir + 'lu.haze.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss',
                                                   verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
mc_cb_model_partly_cloudy = keras.callbacks.ModelCheckpoint(dh.basepath + modeldir + 'lu.partly_cloudy.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss',
                                                            verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
mc_cb_model_clear = keras.callbacks.ModelCheckpoint(dh.basepath + modeldir + 'lu.clear.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss',
                                                    verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

# CNN
model_atmos = keras.models.Sequential()
model_atmos.add(keras.layers.Conv2D(32, kernel_size=(3, 3),
                activation='relu',
                input_shape=input_shape))
model_atmos.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model_atmos.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model_atmos.add(keras.layers.Dropout(0.25))
model_atmos.add(keras.layers.Flatten())
model_atmos.add(keras.layers.Dense(128, activation='relu'))
model_atmos.add(keras.layers.Dropout(0.5))
model_atmos.add(keras.layers.Dense(num_classes_atmos, activation='softmax'))

model_atmos.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adadelta(),
                    metrics=['accuracy'])

model_haze = keras.models.Sequential()
model_haze.add(keras.layers.Conv2D(32, kernel_size=(3, 3),
               activation='relu',
               input_shape=input_shape))
model_haze.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model_haze.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model_haze.add(keras.layers.Dropout(0.25))
model_haze.add(keras.layers.Flatten())
model_haze.add(keras.layers.Dense(128, activation='relu'))
model_haze.add(keras.layers.Dropout(0.5))
model_haze.add(keras.layers.Dense(num_classes_landuse, activation='sigmoid'))

model_haze.compile(loss=keras.losses.binary_crossentropy,
                   optimizer=keras.optimizers.Adadelta(),
                   metrics=['accuracy'])

model_partly_cloudy = keras.models.Sequential()
model_partly_cloudy.add(keras.layers.Conv2D(32, kernel_size=(3, 3),
                        activation='relu',
                        input_shape=input_shape))
model_partly_cloudy.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model_partly_cloudy.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model_partly_cloudy.add(keras.layers.Dropout(0.25))
model_partly_cloudy.add(keras.layers.Flatten())
model_partly_cloudy.add(keras.layers.Dense(128, activation='relu'))
model_partly_cloudy.add(keras.layers.Dropout(0.5))
model_partly_cloudy.add(keras.layers.Dense(num_classes_landuse, activation='sigmoid'))

model_partly_cloudy.compile(loss=keras.losses.binary_crossentropy,
                            optimizer=keras.optimizers.Adadelta(),
                            metrics=['accuracy'])

model_clear = keras.models.Sequential()
model_clear.add(keras.layers.Conv2D(32, kernel_size=(3, 3),
                activation='relu',
                input_shape=input_shape))
model_clear.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model_clear.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model_clear.add(keras.layers.Dropout(0.25))
model_clear.add(keras.layers.Flatten())
model_clear.add(keras.layers.Dense(128, activation='relu'))
model_clear.add(keras.layers.Dropout(0.5))
model_clear.add(keras.layers.Dense(num_classes_landuse, activation='sigmoid'))

model_clear.compile(loss=keras.losses.binary_crossentropy,
                    optimizer=keras.optimizers.Adadelta(),
                    metrics=['accuracy'])

outer_batch_size = 512
inner_batch_size = 128
base_remix_factor = 4
haze_remix_factor = base_remix_factor * 8
partly_cloudy_remix_factor = base_remix_factor * 4
remix_epoch = 2
epochs = 170

for e in range(epochs):
    print('Epoch', e)
    train_iter = dh.get_train_iter_preferred_order()  # One pass through the data == epoch
    x_train_atmos = np.empty(shape=(outer_batch_size, imgwidth, imgheight, numchans), dtype='float32')
    x_train_clear = np.empty(shape=(outer_batch_size, imgwidth, imgheight, numchans), dtype='float32')
    x_train_haze = np.empty(shape=(outer_batch_size, imgwidth, imgheight, numchans), dtype='float32')
    x_train_partly_cloudy = np.empty(shape=(outer_batch_size, imgwidth, imgheight, 4), dtype='float32')
    y_train_atmos = np.empty(shape=(outer_batch_size, num_classes_atmos), dtype='bool')
    y_train_landuse_clear = np.empty(shape=(outer_batch_size, num_classes_landuse), dtype='bool')
    y_train_landuse_haze = np.empty(shape=(outer_batch_size, num_classes_landuse), dtype='bool')
    y_train_landuse_partly_cloudy = np.empty(shape=(outer_batch_size, num_classes_landuse), dtype='bool')
    i, ipc, ih, ic = 0, 0, 0, 0
    for X, Y in train_iter:
        x_train_atmos[i % outer_batch_size] = X
        i += 1
        if Y['cloudy'].all():
            print(Y['name'].values, 'is cloudy')
        elif Y['partly_cloudy'].all():
            print(Y['name'].values, 'is partly cloudy')
            x_train_partly_cloudy[ipc % outer_batch_size] = X
            y_train_landuse_partly_cloudy[ipc % outer_batch_size] = Y.loc[:, landuse].as_matrix()[0]
            ipc += 1
        elif Y['haze'].all():
            print(Y['name'].values, 'is haze')
            x_train_haze[ih % outer_batch_size] = X
            y_train_landuse_haze[ih % outer_batch_size] = Y.loc[:, landuse].as_matrix()[0]
            ih += 1
        elif Y['clear'].all():
            print(Y['name'].values, 'is clear')
            x_train_clear[i % outer_batch_size] = X
            y_train_landuse_clear[ic % outer_batch_size] = Y.loc[:, landuse].as_matrix()[0]
            ic += 1
        y_train_atmos[i % outer_batch_size] = Y.loc[:, atmos].as_matrix()[0]
        if i % outer_batch_size == outer_batch_size - 1:
            batches = 0
            for x_batch, y_batch in train_datagen.flow(x_train_atmos, y_train_atmos, batch_size=inner_batch_size):
                print('Fitting atmos model Batch#', batches)
                model_atmos.fit(x_batch, y_batch, epochs=remix_epoch, verbose=1,
                                validation_split=0.2,
                                callbacks=[tb_cb_model_atmos, mc_cb_model_atmos]
                                )
                batches += 1
                if batches >= outer_batch_size // inner_batch_size * base_remix_factor:
                    break
        if ipc % outer_batch_size == outer_batch_size - 1:
            batches = 0
            for x_batch, y_batch in train_datagen.flow(x_train_partly_cloudy, y_train_landuse_partly_cloudy, batch_size=inner_batch_size):
                print('Fitting landuse model for partly coudy atmos Batch#', batches)
                model_partly_cloudy.fit(x_batch, y_batch, epochs=remix_epoch, verbose=1,
                                        validation_split=0.2,
                                        callbacks=[tb_cb_model_partly_cloudy, mc_cb_model_partly_cloudy]
                                        )
                batches += 1
                if batches >= outer_batch_size // inner_batch_size * partly_cloudy_remix_factor:
                    break
        if ih % outer_batch_size == outer_batch_size - 1:
            batches = 0
            for x_batch, y_batch in train_datagen.flow(x_train_haze, y_train_landuse_haze, batch_size=inner_batch_size):
                print('Fitting landuse model for haze atmos Batch#', batches)
                model_haze.fit(x_batch, y_batch, epochs=remix_epoch, verbose=1,
                               validation_split=0.2,
                               callbacks=[tb_cb_model_haze, mc_cb_model_haze]
                               )
                batches += 1
                if batches >= outer_batch_size // inner_batch_size * haze_remix_factor:
                    break
        if ic % outer_batch_size == outer_batch_size - 1:
            batches = 0
            for x_batch, y_batch in train_datagen.flow(x_train_clear, y_train_landuse_clear, batch_size=inner_batch_size):
                print('Fitting landuse model for clear atmos Batch#', batches)
                model_clear.fit(x_batch, y_batch, epochs=remix_epoch, verbose=1,
                                validation_split=0.2,
                                callbacks=[tb_cb_model_clear, mc_cb_model_clear]
                                )
                batches += 1
                if batches >= outer_batch_size // inner_batch_size * base_remix_factor:
                    break
