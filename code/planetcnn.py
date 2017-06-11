import tensorflow.contrib.keras as keras
from tensorflow.contrib.keras import backend as K
import tensorflow as tf
import numpy as np
from sklearn.metrics import fbeta_score

import logging
import sys
import os
import argparse

from planetutils import DataHandler


ATMOS = ['clear', 'partly_cloudy', 'haze', 'cloudy']
ATMOS_W = [1, 2, 4, 4]
LANDUSE = ['primary', 'agriculture', 'road', 'water', 'cultivation', 'habitation', 'bare_ground',
           'artisinal_mine', 'blooming', 'blow_down', 'selective_logging', 'slash_burn', 'conventional_mine']
LANDUSE_W = [1, 2, 2, 2, 4, 4, 8, 8, 8, 8, 8, 8, 8]

H, W, CHANS = 64, 64, 3
IMG_SHAPE = (W, H, CHANS)

DH = DataHandler()
DH.set_train_labels()

SAMPLES = 100
if SAMPLES is None:
    SAMPLES = DH.train_labels.shape[0]


def unet(nc):
    inputs = keras.layers.Input((H, W, CHANS))
    conv1 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = keras.layers.concatenate([keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = keras.layers.concatenate([keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = keras.layers.concatenate([keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = keras.layers.concatenate([keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    flat = keras.layers.Flatten()(conv9)
    dense = keras.layers.Dense(nc, activation='sigmoid')(flat)

    model = keras.models.Model(inputs=[inputs], outputs=[dense])

    model.compile(optimizer=keras.optimizers.Adam(lr=1e-5), loss=keras.losses.binary_crossentropy)

    return model


def multi_label_cnn(nc):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(
              filters=32,
              kernel_size=(5, 5),
              strides=1,
              padding='valid',
              activation='relu',
              input_shape=IMG_SHAPE))
    model.add(keras.layers.Conv2D(
              filters=64,
              kernel_size=(5, 5),
              activation='relu',
              strides=1,
              padding='valid'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(nc, activation='sigmoid'))

    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer='adam')
    return model


class Modeler(object):

    def __init__(self, name, model_f, nc, cw, sample_num):
        self.name = name
        self.model = model_f(nc=nc)
        self.class_weight = cw
        self.sample_num = sample_num
        self.datagen = self.train_datagen()
        self.tb_cb = self.tensorboard_cb()
        self.cp_cb = self.checkpoint_cb()
        self.x_train = np.empty(shape=(self.sample_num, W, H, CHANS), dtype='float32')
        self.y_train = np.empty(shape=(self.sample_num, nc), dtype='bool')
        self.sample_counter = 0

    def __repr__(self):
        return self.name

    # Setup tensorboard callbacks
    def tensorboard_cb(self, basepath=DH.basepath):
        graphdir = basepath + "/graph/" + self.name + "/"
        mkdir(graphdir)
        return keras.callbacks.TensorBoard(log_dir=graphdir,
                                           histogram_freq=0,
                                           write_graph=False,
                                           write_images=False)

    # Setup model checkpoint callbacks
    def checkpoint_cb(self, basepath=DH.basepath):
        modeldir = basepath + "/model/" + self.name + "/"
        mkdir(modeldir)
        cp_fn = "{epoch:03d}-{val_loss:.5f}.hdf5"
        return keras.callbacks.ModelCheckpoint(modeldir + cp_fn, monitor='val_loss',
                                               save_best_only=False,
                                               save_weights_only=False,
                                               mode='auto',
                                               period=1)

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

    def checkpoint(self):
        LOG.info("Saving model checkpoint for [%s]" % self.name)
        cp_fn = "%s.hdf5" % self.name
        keras.models.save_model(self.model, DH.basepath + "/model/" + cp_fn)
        return


def load_model(mfn):
    mfn += ".hdf5"
    keras.backend.manual_variable_initialization(True)
    path = DH.basepath + "/model-archive/" + mfn
    M = keras.models.load_model(path)
    return M


def prediction(M, X, thresh):
    model_prediction = M.predict(np.array([X]))
    model_prediction = model_prediction[0]
    labels = ATMOS + LANDUSE
    for i, elem in enumerate(model_prediction):
        model_prediction[i] = model_prediction[i] > thresh[i]
    result = []
    for i, label in enumerate(labels):
        if model_prediction[i]:
            result.append(label)
    return " ".join(result)


def fbeta(true_label, prediction):
    return fbeta_score(true_label, prediction, beta=2, average='samples')


def get_optimal_threshhold(true_label, prediction, iterations=100):
    best_threshhold = [0.2] * 17
    for t in range(17):
        best_fbeta = 0
        temp_threshhold = [0.2] * 17
        for i in range(iterations):
            temp_value = i / float(iterations)
            temp_threshhold[t] = temp_value
            temp_fbeta = fbeta(true_label, prediction > temp_threshhold)
            if temp_fbeta > best_fbeta:
                best_fbeta = temp_fbeta
                best_threshhold[t] = temp_value
    return best_threshhold


def calc_thresh(m, imgtyp):
    LOG.info("Optimizing thresholds")
    true_label, p = [], []
    for X, Y in DH.get_train_iter(imgtyp=imgtyp, h=H, w=W, maxn=SAMPLES // 5):
        Y = Y.loc[:, ATMOS + LANDUSE].as_matrix()[0]
        true_label.append(list(Y))
        X = X / (np.power(2, 16) - 1.)
        pred = m.predict(np.array([X]))
        p.append(list(pred[0]))
    t = get_optimal_threshhold(np.array(true_label), np.array(p))
    LOG.info("Best thresholds for %s are %s" % (ATMOS + LANDUSE, t))
    return t


def write_submission(outputpath, m, thresh, imgtyp):
    maxn = None if SAMPLES > 10000 else 100
    with open(DH.basepath + outputpath, "w") as fp:
        fp.write("image_name,tags\n")
        for name, X in DH.get_test_iter(imgtyp=imgtyp, h=H, w=W, maxn=maxn):
            X = X / (np.power(2, 16) - 1.)
            p = prediction(m, X, thresh)
            fp.write("%s,%s\n" % (name, p))


def mkdir(d):
    if not os.path.exists(d):
        os.mkdir(d)


def main():
    # name = "64x64-32-5x5-64-5x5-jpg"
    name = "base-unet-jpg"
    imgtyp = "jpg"
    M = Modeler(name, unet, len(ATMOS) + len(LANDUSE), ATMOS_W + LANDUSE_W, SAMPLES)
    submission = "%s.csv" % name
    if args.train:
        LOG.info("Starting training run")
        for X, Y in DH.get_train_iter(imgtyp=imgtyp, h=H, w=W, maxn=SAMPLES):
            M.set_x_y(X, Y.loc[:, ATMOS + LANDUSE].as_matrix()[0])
        epochs = 50
        try:
            M.fit_full_datagen(epochs=epochs)
            thresh = calc_thresh(M.model, imgtyp)
            write_submission("/output/%s" % submission, M.model, thresh, imgtyp)
        except KeyboardInterrupt:
            LOG.info("Stopping training and checkpointing the model")
            M.checkpoint()
            thresh = calc_thresh(M.model, imgtyp)
            write_submission("/output/%s" % submission, M.model, thresh, imgtyp)
    elif args.test:
        m = load_model(name)
        thresh = calc_thresh(m, imgtyp)
        write_submission("/output/%s" % submission, m, thresh, imgtyp)
    elif args.thresh:
        m = load_model(name)
        calc_thresh(m, imgtyp)


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
    logfile = DH.basepath + "/log/" + "planet-kaggle-cnn-v4.log"
    handler = logging.FileHandler(logfile)
    formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    LOG.addHandler(handler)
    sys.stdout = StreamToLogger(LOG)
    sys.stderr = sys.stdout

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-on-interrupt', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--thresh', action='store_true')
    args = parser.parse_args()
    main()
