import tensorflow.contrib.keras as keras
from tensorflow.contrib.keras import backend as K
import tensorflow as tf
import numpy as np
from sklearn.metrics import fbeta_score

import logging
import sys
import argparse

from planetutils import DataHandler


ATMOS = ['clear', 'partly_cloudy', 'haze', 'cloudy']
ATMOS_W = [1, 2, 4, 4]
LANDUSE = ['primary', 'agriculture', 'road', 'water', 'cultivation', 'habitation', 'bare_ground',
           'artisinal_mine', 'blooming', 'blow_down', 'selective_logging', 'slash_burn', 'conventional_mine']
LANDUSE_W = [1, 2, 2, 2, 4, 4, 8, 8, 8, 8, 8, 8, 8]

H, W, CHANS = 64, 64, 4
IMG_SHAPE = (W, H, CHANS)

DH = DataHandler()
DH.set_train_labels()


def multi_label_cnn(nc=len(LANDUSE)):
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

    def checkpoint(self):
        LOG.info("Saving model checkpoint for [%s]" % self.name)
        cp_fn = "%s.hdf5" % self.name
        keras.models.save_model(self.model, DH.basepath + "/model/" + cp_fn)
        return


def load_model(mfn="20170607-all/all-297-0.15.hdf5"):
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


def calc_thresh(m):
    LOG.info("Optimizing thresholds")
    true_label, p = [], []
    i = 0
    for X, Y in DH.get_train_iter(h=H, w=W):
        Y = Y.loc[:, ATMOS + LANDUSE].as_matrix()[0]
        true_label.append(list(Y))
        X = X / (np.power(2, 16) - 1.)
        pred = m.predict(np.array([X]))
        p.append(list(pred[0]))
        i += 1
        if i % 1000 == 0:
            LOG.info("%s samples predicted" % i)
        if i == 20000:
            break
    t = get_optimal_threshhold(np.array(true_label), np.array(p))
    LOG.info("Best thresholds for %s are %s" % (ATMOS + LANDUSE, t))
    return t


def write_submission(outputpath, m, thresh):
    with open(DH.basepath + outputpath, "w") as fp:
        fp.write("image_name,tags\n")
        for name, X in DH.get_test_iter(h=H, w=W):
            X = X / (np.power(2, 16) - 1.)
            p = prediction(m, X, thresh)
            fp.write("%s,%s\n" % (name, p))


def main():
    name = "64x64-32-5x5-64-5x5"
    M = Modeler(name, multi_label_cnn, len(ATMOS) + len(LANDUSE), ATMOS_W + LANDUSE_W)
    submission = "%s.csv" % name
    if args.train:
        LOG.info("Starting training run")
        for X, Y in DH.get_train_iter(h=H, w=W):
            M.set_x_y(X, Y.loc[:, ATMOS + LANDUSE].as_matrix()[0])
        epochs = 50
        try:
            M.fit_full_datagen(epochs=epochs)
            thresh = calc_thresh(M.model)
            write_submission("/output/%s" % submission, M.model, thresh)
        except KeyboardInterrupt:
            LOG.info("Stopping training and checkpointing the model")
            M.checkpoint()
            thresh = calc_thresh(M.model)
            write_submission("/output/%s" % submission, M.model, thresh)
    elif args.test:
        m = load_model()
        thresh = calc_thresh(m)
        write_submission("/output/%s" % submission, m, thresh)

    elif args.thresh:
        m = load_model()
        calc_thresh(m)


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
