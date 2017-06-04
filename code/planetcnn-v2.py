import tensorflow.contrib.keras as keras
from tensorflow.contrib.keras import backend as K
import numpy as np
import logging
import sys

from planetutils import DataHandler


DH = DataHandler()
DH.set_train_labels()
ATMOS = ['cloudy', 'partly_cloudy', 'haze', 'clear']
LANDUSE = ['artisinal_mine', 'blooming', 'blow_down', 'agriculture', 'bare_ground',
           'primary', 'road', 'selective_logging', 'slash_burn', 'water',
           'conventional_mine', 'cultivation', 'habitation']

W, H, CHANS = 256, 256, 4
IMG_SHAPE = (W, H, CHANS)


def single_label_cnn(nc=len(ATMOS)):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, kernel_size=(3, 3),
              activation='relu',
              input_shape=IMG_SHAPE,
              name="Conv2D-1"))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', name="Conv2D-2"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), name="MaxPooling2D"))
    model.add(keras.layers.Dropout(0.25, name="Dropout-1"))
    model.add(keras.layers.Flatten(name="Flatten"))
    model.add(keras.layers.Dense(128, activation='relu', name="Dense-relu"))
    model.add(keras.layers.Dropout(0.5, name="Dropout-2"))
    model.add(keras.layers.Dense(nc, activation='softmax', name="Dense-softmax"))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model


def multi_label_cnn(nc=len(LANDUSE)):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, kernel_size=(3, 3),
              activation='relu',
              input_shape=IMG_SHAPE,
              name="Conv2D-1"))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', name="Conv2D-2"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), name="MaxPooling2D"))
    model.add(keras.layers.Dropout(0.25, name="Dropout-1"))
    model.add(keras.layers.Flatten(name="Flatten"))
    model.add(keras.layers.Dense(128, activation='relu', name="Dense-relu"))
    model.add(keras.layers.Dropout(0.5, name="Dropout-2"))
    model.add(keras.layers.Dense(nc, activation='sigmoid', name="Dense-sigmoid"))

    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model


class Modeler(object):
    outer_batch_size = 512
    inner_batch_size = 256
    base_remix_factor = 4
    remix_epoch = 2

    def __init__(self, name, model_f, nc):
        self.name = name
        self.model = model_f()
        self.batch_counter = 0
        self.epoch_counter = 1
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
                                           write_graph=True, write_images=True)

    # Setup model checkpoint callbacks
    def checkpoint_cb(self, model, basepath=DH.basepath):
        modeldir = basepath + "/model/" + model
        cp_fn = "-".join([self.name, "epoch:%s" % self.epoch_counter, "{val_loss:.2f}"]) + ".hdf5"
        return keras.callbacks.ModelCheckpoint(modeldir + cp_fn, monitor='val_loss',
                                               verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

    def train_datagen(self):
        return keras.preprocessing.image.ImageDataGenerator(
            rescale=1. / (np.power(2, 16) - 1),
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    def set_x_y(self, x, y):
        self.x_train[self.batch_counter % self.outer_batch_size] = x
        self.y_train[self.batch_counter % self.outer_batch_size] = y
        self.batch_counter += 1

    def do_fit(self):
        return self.batch_counter % self.outer_batch_size == self.outer_batch_size - 1

    def fit_minibatch(self):
        batches = 0
        for x_batch, y_batch in self.datagen.flow(self.x_train, self.y_train, self.inner_batch_size):
            LOG.info('Fitting %s model batch %s' % (self.name, batches))
            self.model.fit(x_batch, y_batch,
                           epochs=self.remix_epoch,
                           verbose=2,
                           validation_split=0.2,
                           callbacks=[self.tb_cb]
                           )
            batches += 1
            if batches >= self.outer_batch_size // self.inner_batch_size * self.base_remix_factor * self.class_balance_remix_factor:
                break
        return

    def checkpoint(self):
        LOG.info("Saving model checkpoint for [%s]" % self.name)
        self.model.fit(self.x_train, self.y_train,
                       epochs=1,
                       verbose=0,
                       validation_split=0.2,
                       callbacks=[self.tb_cb, self.cp_cb]
                       )
        return


def main():
    # Setup models
    M_atmos = Modeler("atmos", single_label_cnn, len(ATMOS))
    M_clear = Modeler("clear", multi_label_cnn, len(LANDUSE))
    M_haze = Modeler("haze", multi_label_cnn, len(LANDUSE))
    M_partly_cloudy = Modeler("partly-cloudy", multi_label_cnn, len(LANDUSE))

    epochs = 10
    for e in range(epochs):
        LOG.info('Epoch %s' % e)
        train_iter = DH.get_train_iter()  # One pass through the data == epoch
        for X, Y in train_iter:
            M_atmos.set_x_y(X, Y.loc[:, ATMOS].as_matrix()[0])
            if Y['cloudy'].all():
                continue
            elif Y['partly_cloudy'].all():
                M_partly_cloudy.set_x_y(X, Y.loc[:, LANDUSE].as_matrix()[0])
            elif Y['haze'].all():
                M_haze.set_x_y(X, Y.loc[:, LANDUSE].as_matrix()[0])
            elif Y['clear'].all():
                M_clear.set_x_y(X, Y.loc[:, LANDUSE].as_matrix()[0])
            if M_atmos.do_fit():
                M_atmos.fit_minibatch()
            if M_clear.do_fit():
                M_clear.fit_minibatch()
            elif M_haze.do_fit():
                M_haze.fit_minibatch()
            elif M_partly_cloudy.do_fit():
                M_partly_cloudy.fit_minibatch()
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
    main()
