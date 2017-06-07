import tensorflow.contrib.keras as keras
from tensorflow.contrib.keras import backend as K
import tensorflow as tf
import numpy as np
import logging
import sys
import argparse

from planetutils import DataHandler


DH = DataHandler()
DH.set_train_labels()
ATMOS = ['clear', 'partly_cloudy', 'haze', 'cloudy']
ATMOS_W = [1, 2, 4, 4]
LANDUSE = ['primary', 'agriculture', 'road', 'water', 'cultivation', 'habitation', 'bare_ground',
           'artisinal_mine', 'blooming', 'blow_down', 'selective_logging', 'slash_burn', 'conventional_mine']
LANDUSE_W = [1, 2, 2, 2, 4, 4, 8, 8, 8, 8, 8, 8, 8]

H, W, CHANS = 32, 32, 4
IMG_SHAPE = (W, H, CHANS)


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
    outer_batch_size = 8192
    inner_batch_size = 4096
    mini_batch_size = 32
    base_remix_factor = 4
    remix_epoch = 1

    def __init__(self, name, model_f, nc, cw):
        self.name = name
        self.model = model_f()
        self.batch_counter = 0
        self.batch_counter_lag_one = -1
        self.epoch_counter = 1
        self.class_weight = cw
        self.datagen = self.train_datagen()
        self.tb_cb = self.tensorboard_cb(self.name)
        self.cp_cb = self.checkpoint_cb(self.name)
        self.x_train = np.empty(shape=(self.outer_batch_size, W, H, CHANS), dtype='float32')
        self.y_train = np.empty(shape=(self.outer_batch_size, nc), dtype='bool')
        self.class_balance_remix_factor = self.set_class_balance_remix_factor()

    def __repr__(self):
        return self.name

    def set_class_balance_remix_factor(self):
        res = 1
        if self.name == 'clear':
            res = 2
        elif self.name == 'partly_cloudy':
            res = 4
        elif self.name == 'haze':
            res = 8
        return res

    # Setup tensorboard callbacks
    def tensorboard_cb(self, model, basepath=DH.basepath):
        graphdir = basepath + "/graph/" + model
        return keras.callbacks.TensorBoard(log_dir=graphdir, histogram_freq=0,
                                           write_graph=False, write_images=False)

    # Setup model checkpoint callbacks
    def checkpoint_cb(self, model, basepath=DH.basepath):
        modeldir = basepath + "/model/" + model
        cp_fn = "-".join([self.name, "epoch:%s" % self.epoch_counter, "{val_loss:.2f}"]) + ".hdf5"
        return keras.callbacks.ModelCheckpoint(modeldir + cp_fn, monitor='val_loss',
                                               verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

    def train_datagen(self):
        return keras.preprocessing.image.ImageDataGenerator(
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    def set_x_y(self, x, y):
        self.x_train[self.batch_counter % self.outer_batch_size] = x / (np.pow(2, 16) - 1.)
        self.y_train[self.batch_counter % self.outer_batch_size] = y
        self.batch_counter += 1
        self.batch_counter_lag_one += 1

    def do_fit(self):
        return self.batch_counter_lag_one % self.outer_batch_size == self.outer_batch_size - 1

    def fit_minibatch_with_datagen(self):
        batches = 0
        for x_batch, y_batch in self.datagen.flow(self.x_train, self.y_train, self.inner_batch_size):
            self.fit(x_batch, y_batch)
            batches += 1
            if batches >= self.outer_batch_size // self.inner_batch_size * self.base_remix_factor * self.class_balance_remix_factor:
                break
        return

    def fit_minibatch(self):
        self.fit(self.x_train, self.y_train)
        return

    def fit(self, x, y):
        LOG.info('Fitting %s model: %s total images examined after %s epochs' % (self.name, self.batch_counter + 1, self.epoch_counter))
        self.model.fit(x, y,
                       epochs=self.remix_epoch,
                       verbose=2,
                       validation_split=0.2,
                       # class_weight=self.class_weight,
                       shuffle=False,
                       batch_size=self.mini_batch_size,
                       callbacks=[self.tb_cb]
                       )

    def checkpoint(self):
        LOG.info("Saving model checkpoint for [%s]" % self.name)
        cp_fn = "-".join([self.name, "epoch:%s" % self.epoch_counter]) + ".hdf5"
        keras.models.save_model(self.model, DH.basepath + "/model/" + cp_fn)
        return


def test_model_serialization(m, basepath=DH.basepath):
    modeldir = basepath + "/model/"
    cp_fn = "test-serialization.hdf5"
    m_saved_weights = m.get_weights()
    keras.models.save_model(m, modeldir + cp_fn)
    K.clear_session()
    m_load = keras.models.load_model(modeldir + cp_fn)
    m_load_weights = m_load.get_weights()
    assert len(m_load_weights) == len(m_saved_weights)
    for i in range(len(m_saved_weights)):
        LOG.info(m_load_weights[i].shape)
        LOG.info(m_saved_weights[i].shape)
        assert np.array_equal(m_load_weights[i], m_saved_weights[i])
    LOG.info("Serialization test passed")
    return m_load


def main():
    epochs = 10
    for e in range(epochs):
        LOG.info('Epoch %s' % e)
        train_iter = DH.get_train_iter(h=H, w=W)  # One pass through the data == epoch
        for X, Y in train_iter:
            # Training an atmospheric model
            M_atmos.set_x_y(X, Y.loc[:, ATMOS].as_matrix()[0])
            if M_atmos.do_fit():
                M_atmos.fit_minibatch()

            # Trainging landuse models for various atmospheric conditions
            if Y['cloudy'].all():
                continue
            elif Y['partly_cloudy'].all():
                M_partly_cloudy.set_x_y(X, Y.loc[:, LANDUSE].as_matrix()[0])
                if M_partly_cloudy.do_fit():
                    M_partly_cloudy.fit_minibatch()
            elif Y['haze'].all():
                M_haze.set_x_y(X, Y.loc[:, LANDUSE].as_matrix()[0])
                if M_haze.do_fit():
                    M_haze.fit_minibatch()
            elif Y['clear'].all():
                M_clear.set_x_y(X, Y.loc[:, LANDUSE].as_matrix()[0])
                if M_clear.do_fit():
                    M_clear.fit_minibatch()
        M_atmos.checkpoint()
        M_atmos.epoch_counter += 1
        M_clear.checkpoint()
        M_clear.epoch_counter += 1
        M_haze.checkpoint()
        M_haze.epoch_counter += 1
        M_partly_cloudy.checkpoint()
        M_partly_cloudy.epoch_counter += 1


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
