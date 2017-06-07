import tensorflow.contrib.keras as keras
from tensorflow.contrib.keras import backend as K
import tensorflow as tf
import numpy as np
import logging
import sys
import argparse

from planetutils import DataHandler


ATMOS = ['clear', 'partly_cloudy', 'haze', 'cloudy']
ATMOS_W = [1, 2, 4, 4]
LANDUSE = ['primary', 'agriculture', 'road', 'water', 'cultivation', 'habitation', 'bare_ground',
           'artisinal_mine', 'blooming', 'blow_down', 'selective_logging', 'slash_burn', 'conventional_mine']
LANDUSE_W = [1, 2, 2, 2, 4, 4, 8, 8, 8, 8, 8, 8, 8]

H, W, CHANS = 32, 32, 4
IMG_SHAPE = (W, H, CHANS)

DH = DataHandler()
DH.set_train_labels()


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
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(nc, activation='sigmoid'))

    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


class Modeler(object):
    sample_num = 40479

    def __init__(self, name, model_f, nc, cw):
        self.name = name
        self.model = model_f(nc=nc)
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
        modeldir = basepath + "/model/"
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


def load_model(mfn="20170607-all/all-297-0.15.hdf5"):
    path = DH.basepath + "/model-archive/" + mfn
    M = keras.models.load_model(path)
    return M


def prediction(M, X):
    p = M.predict(np.array([X]))
    labels = ATMOS + LANDUSE
    simple_eval = lambda x: 1 if x > 0.5 else 0
    p = map(simple_eval, p[0])
    result = []
    for i, label in enumerate(labels):
        if p[i] == 1:
            result.append(label)
    return " ".join(result)


def main():
    if not args.test:
        for X, Y in DH.get_train_iter(h=H, w=W):
            M.set_x_y(X, Y.loc[:, ATMOS + LANDUSE].as_matrix()[0])
        epochs = 500
        M.fit_full_datagen(epochs=epochs)
    m = load_model()
    with open(DH.basepath + "/output/20170607-all/submit.csv", "w") as fp:
        fp.write("image_name,tags\n")
        for name, X in DH.get_test_iter(h=H, w=W):
            X = X / (np.power(2, 16) - 1.)
            p = prediction(m, X)
            fp.write("%s,%s\n" % (name, p))


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
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    # Setup models
    M = Modeler("all", multi_label_cnn, len(ATMOS) + len(LANDUSE), ATMOS_W + LANDUSE_W)

    try:
        main()
    except KeyboardInterrupt:
        if args.checkpoint_on_interrupt:
            M.checkpoint()
