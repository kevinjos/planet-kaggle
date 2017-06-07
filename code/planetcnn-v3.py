import tensorflow.contrib.keras as keras
from tensorflow.contrib.keras import backend as K
import tensorflow as tf
import numpy as np
import logging
import sys
import argparse

from planetutils import DataHandler

import os
from spectral import ndvi, get_rgb
from skimage import io
from sklearn.preprocessing import MinMaxScaler
from PIL import Image

import matplotlib.pyplot as plt


ATMOS = ['clear', 'partly_cloudy', 'haze', 'cloudy']
ATMOS_W = [1, 2, 4, 4]
LANDUSE = ['primary', 'agriculture', 'road', 'water', 'cultivation', 'habitation', 'bare_ground',
           'artisinal_mine', 'blooming', 'blow_down', 'selective_logging', 'slash_burn', 'conventional_mine']
LANDUSE_W = [1, 2, 2, 2, 4, 4, 8, 8, 8, 8, 8, 8, 8]

H, W, CHANS = 32, 32, 4
IMG_SHAPE = (W, H, CHANS)

DH = DataHandler()
DH.set_train_labels()


def load_model(modelpath="/model-archive/20170606-atmos/atmosatmos-91-0.36.hdf5"):
    return keras.models.load_model(DH.basepath + modelpath)


def inspect(m=load_model()):
    data_iter = DH.get_manual_train_iter(h=H, w=W)
    right = dict(zip(ATMOS, [0 for x in range(len(ATMOS))]))
    wrong = dict(zip(ATMOS, [dict(zip(ATMOS, [[] for x in range(len(ATMOS))])) for y in range(len(ATMOS))]))
    wrong["names"] = []
    for name, x, y in data_iter:
        x = x / (np.power(2, 16) - 1.)
        y_predict = m.predict(np.array([x]))
        y = y.loc[:, ATMOS].as_matrix()[0]
        prediction = ATMOS[np.argmax(y_predict)]
        actual = ATMOS[np.argmax(y)]
        if prediction == actual:
            right[actual] += 1
        else:
            wrong[actual][prediction].append(name)
    return right, wrong


def predict(fn, m):
    x = DH.get_tif(fn, h=H, w=W)
    x = x / (np.power(2, 16) - 1.)
    return ATMOS[np.argmax(m.predict(np.array([x])))]


def actual(fn):
    name = fn.split(".")[0]
    return ATMOS[np.argmax(DH.train_labels.loc[DH.train_labels["name"] == name].loc[:, ATMOS].as_matrix()[0])]


def view(name):
    path = os.path.join(DH.basepath + "/input/train-tif-v2/", name)
    img = io.imread(path)
    img2 = get_rgb(img, [2, 1, 0])
    img3 = get_rgb(img, [3, 2, 1])
    img4 = get_rgb(img, [3, 2, 0])

    # rescaling to 0-255 range - uint8 for display
    rescaleIMG = np.reshape(img2, (-1, 1))
    scaler = MinMaxScaler(feature_range=(0, 255))
    rescaleIMG = scaler.fit_transform(rescaleIMG)
    img2_scaled = (np.reshape(rescaleIMG, img2.shape)).astype(np.uint8)

    rescaleIMG = np.reshape(img3, (-1, 1))
    scaler = MinMaxScaler(feature_range=(0, 255))
    rescaleIMG = scaler.fit_transform(rescaleIMG)
    img3_scaled = (np.reshape(rescaleIMG, img3.shape)).astype(np.uint8)

    rescaleIMG = np.reshape(img4, (-1, 1))
    scaler = MinMaxScaler(feature_range=(0, 255))
    rescaleIMG = scaler.fit_transform(rescaleIMG)
    img4_scaled = (np.reshape(rescaleIMG, img4.shape)).astype(np.uint8)

    # spectral module ndvi function
    vi = ndvi(img, 2, 3)

    # calculate NDVI and NDWI with spectral module adjusted bands
    np.seterr(all='warn')
    vi2 = (img3[:, :, 0] - img3[:, :, 1]) / (img3[:, :, 0] + img3[:, :, 1])
    vi3 = (img3[:, :, 2] - img3[:, :, 0]) / (img3[:, :, 2] + img3[:, :, 0])

    plt.style.use('ggplot')
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 10))
    ax = axes.ravel()
    ax[0] = plt.subplot(2, 3, 1, adjustable='box-forced')
    ax[1] = plt.subplot(2, 3, 2, sharex=ax[0], sharey=ax[0], adjustable='box-forced')
    ax[2] = plt.subplot(2, 3, 3, sharex=ax[0], sharey=ax[0], adjustable='box-forced')
    ax[3] = plt.subplot(2, 3, 4, adjustable='box-forced')
    ax[4] = plt.subplot(2, 3, 5, adjustable='box-forced')
    ax[5] = plt.subplot(2, 3, 6, adjustable='box-forced')
    ax[0].imshow(img2_scaled)
    ax[0].set_title('RGB')
    ax[1].imshow(img3_scaled)
    ax[1].set_title('NIR-RED-GREEN')
    ax[2].imshow(img4_scaled)
    ax[2].set_title('NIR-RED-BLUE')

    # alternative cmaps e.g. nipy_spectral, gist_earth, terrain
    ax[3].imshow(vi, cmap=plt.get_cmap('nipy_spectral'))
    ax[3].set_title('NDVI-spectral func')
    ax[4].imshow(vi2, cmap=plt.get_cmap('nipy_spectral'))  # , cmap=plt.cm.gray)
    ax[4].set_title('NDVI-calculated')
    ax[5].imshow(vi3, cmap=plt.get_cmap('nipy_spectral'))  # , cmap=plt.cm.gray)
    ax[5].set_title('NDWI GREEN-NIR')
    plt.show()


def single_label_cnn(nc=len(ATMOS)):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(
              filters=32,
              kernel_size=(8, 8),
              strides=1,
              padding='valid',
              activation='relu',
              input_shape=IMG_SHAPE))
    model.add(keras.layers.Conv2D(
              filters=64,
              kernel_size=(8, 8),
              strides=1,
              padding='valid',
              activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(nc, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model


def multi_label_cnn(nc=len(LANDUSE)):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(
              filters=32,
              kernel_size=(8, 8),
              strides=1,
              padding='valid',
              activation='relu',
              input_shape=IMG_SHAPE))
    model.add(keras.layers.Conv2D(
              filters=64,
              kernel_size=(8, 8),
              activation='relu',
              strides=1,
              padding='valid'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(nc, activation='sigmoid'))

    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


class Modeler(object):
    sample_num = 40479

    def __init__(self, name, model_f, nc, cw):
        self.name = name
        self.model = model_f()
        self.class_weight = cw
        self.datagen = self.train_datagen()
        self.tb_cb = self.tensorboard_cb(self.name)
        self.cp_cb = self.checkpoint_cb(self.name)
        self.x_train = np.empty(shape=(self.sample_num, W, H, CHANS), dtype='float32')
        self.y_train = np.empty(shape=(self.sample_num, nc), dtype='bool')
        self.sample_counter = 0

    def __repr__(self):
        return self.name

    # Setup tensorboard callbacks
    def tensorboard_cb(self, model, basepath=DH.basepath):
        graphdir = basepath + "/graph/" + model
        return keras.callbacks.TensorBoard(log_dir=graphdir,
                                           histogram_freq=0,
                                           write_graph=False,
                                           write_images=False)

    # Setup model checkpoint callbacks
    def checkpoint_cb(self, model, basepath=DH.basepath):
        modeldir = basepath + "/model/" + model
        cp_fn = "-".join([self.name, "{epoch:02d}-{val_loss:.2f}"]) + ".hdf5"
        return keras.callbacks.ModelCheckpoint(modeldir + cp_fn, monitor='val_loss',
                                               verbose=1,
                                               save_best_only=False,
                                               save_weights_only=False,
                                               mode='auto',
                                               period=2)

    def train_datagen(self):
        return keras.preprocessing.image.ImageDataGenerator(
            shear_range=0.2,
            zoom_range=0.2,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True)

    def set_x_y(self, x, y):
        self.x_train[self.sample_counter % self.sample_num] = x / (np.power(2, 16) - 1.)
        self.y_train[self.sample_counter % self.sample_num] = y
        self.sample_counter += 1

    def fit_full_datagen(self, epochs):
        samples_total = len(self.y_train)
        split = int(samples_total * .2)
        samples = samples_total - split
        x_val, y_val = self.x_train[:split], self.y_train[:split]
        x_train, y_train = self.x_train[split:], self.y_train[split:]
        batch_size = 32
        steps_per_epochs = samples // batch_size
        LOG.info("Training with data generation for model=[%s]" % self.name)
        LOG.info("train samples=[%s], validation samples=[%s], batch size=[%s], epochs=[%s]" % (samples, split, batch_size, epochs))
        self.model.fit_generator(self.datagen.flow(x_train, y_train, batch_size=batch_size),
                                 steps_per_epochs,
                                 epochs=epochs,
                                 verbose=2,
                                 validation_data=(x_val, y_val),
                                 callbacks=[self.tb_cb, self.cp_cb])


def main():
    for X, Y in DH.get_train_iter(h=H, w=W):
        # Training an atmospheric model
        if args.model == "atmos":
            M_atmos.set_x_y(X, Y.loc[:, ATMOS].as_matrix()[0])
            continue

        # Trainging landuse models for various atmospheric conditions
        if Y['cloudy'].all():
            continue
        elif Y['partly_cloudy'].all() and args.model == "partly_cloudy":
            M_partly_cloudy.set_x_y(X, Y.loc[:, LANDUSE].as_matrix()[0])
        elif Y['haze'].all() and args.model == "haze":
            M_haze.set_x_y(X, Y.loc[:, LANDUSE].as_matrix()[0])
        elif Y['clear'].all() and args.model == "clear":
            M_clear.set_x_y(X, Y.loc[:, LANDUSE].as_matrix()[0])
    epochs = 100
    if args.model == "atmos":
        M_atmos.fit_full_datagen(epochs=epochs)
    elif args.model == "partly_cloudy":
        M_partly_cloudy.fit_full_datagen(epochs=epochs)
    elif args.model == "haze":
        M_haze.fit_full_datagen(epochs=epochs)
    elif args.model == "clear":
        M_clear.fit_full_datagen(epochs=epochs)


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

if __name__ == '__main__':
    LOG = logging.getLogger(__name__)
    LOG.setLevel(logging.INFO)
    logfile = DH.basepath + "/log/" + "planet-kaggle-cnn-v2.log"
    handler = logging.FileHandler(logfile)
    formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    LOG.addHandler(handler)
    LOG.info("Starting training run")
    sys.stdout = StreamToLogger(LOG)
    sys.stderr = sys.stdout

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-on-interrupt', action='store_true')
    parser.add_argument('--model', type=str, help="values include: [atmos, haze, clear, partly_cloudy]")
    args = parser.parse_args()

    # Setup models
    M_atmos = Modeler("atmos", single_label_cnn, len(ATMOS), ATMOS_W)
    M_clear = Modeler("clear", multi_label_cnn, len(LANDUSE), ATMOS_W)
    M_haze = Modeler("haze", multi_label_cnn, len(LANDUSE), ATMOS_W)
    M_partly_cloudy = Modeler("partly-cloudy", multi_label_cnn, len(LANDUSE), ATMOS_W)

    try:
        main()
    except KeyboardInterrupt:
        if args.checkpoint_on_interrupt:
            M_atmos.checkpoint()
            M_clear.checkpoint()
            M_haze.checkpoint()
            M_partly_cloudy.checkpoint()
