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


ATMOS = ['clear',
         'partly_cloudy',
         'haze', 'cloudy']
ATMOS_W = [1, 2, 4, 4]
LANDUSE = ['primary', 'agriculture', 'road', 'water',
           'cultivation', 'habitation',
           'bare_ground', 'artisinal_mine', 'blooming', 'blow_down', 'selective_logging', 'slash_burn', 'conventional_mine']
LANDUSE_W = [1, 2, 2, 2, 4, 4, 8, 8, 8, 8, 8, 8, 8]

H, W, CHANS = 64, 64, 4
IMG_SHAPE = (W, H, CHANS)

DH = DataHandler()
DH.set_train_labels()

SAMPLES = None
if SAMPLES is None:
    SAMPLES = DH.train_labels.shape[0]


def pretrained(nc):
    # input_tensor = keras.layers.Input(batch_shape=(None, 64, 64, 3))
    # vgg = keras.applications.VGG16(input_tensor=input_tensor, input_shape=IMG_SHAPE, weights='imagenet', include_top=False)
    vgg = keras.applications.VGG16(weights='imagenet', include_top=False)
    for layer in vgg.layers:
        layer.trainable = False
    x = vgg.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(4096, activation='relu')(x)
    x = keras.layers.Dense(2048, activation='relu')(x)
    x = keras.layers.Dense(1024, activation='relu')(x)
    x = keras.layers.Dense(512, activation='relu')(x)
    predictions = keras.layers.Dense(nc, activation='sigmoid')(x)
    model = keras.models.Model(inputs=vgg.input, outputs=predictions)
    model.compile(keras.optimizers.SGD(lr=1e-4, momentum=0.9),
                  keras.losses.binary_crossentropy,
                  metrics=['accuracy'])
    return model


def unet(nc):
    inputs = keras.layers.Input((H, W, CHANS))
    conv1 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
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
              kernel_size=(3, 3),
              activation='relu',
              strides=1,
              padding='valid',
              input_shape=IMG_SHAPE))
    model.add(keras.layers.Conv2D(
              filters=32,
              kernel_size=(3, 3),
              activation='relu',
              strides=1,
              padding='valid'))
    model.add(keras.layers.Conv2D(
              filters=32,
              kernel_size=(3, 3),
              activation='relu',
              strides=1,
              padding='valid'))
    model.add(keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Conv2D(
              filters=64,
              kernel_size=(3, 3),
              activation='relu',
              strides=1,
              padding='valid',
              input_shape=IMG_SHAPE))
    model.add(keras.layers.Conv2D(
              filters=64,
              kernel_size=(3, 3),
              activation='relu',
              strides=1,
              padding='valid'))
    model.add(keras.layers.Conv2D(
              filters=64,
              kernel_size=(3, 3),
              activation='relu',
              strides=1,
              padding='valid'))
    model.add(keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Conv2D(
              filters=128,
              kernel_size=(3, 3),
              activation='relu',
              strides=1,
              padding='valid',
              input_shape=IMG_SHAPE))
    model.add(keras.layers.Conv2D(
              filters=128,
              kernel_size=(3, 3),
              activation='relu',
              strides=1,
              padding='valid'))
    model.add(keras.layers.Conv2D(
              filters=128,
              kernel_size=(3, 3),
              activation='relu',
              strides=1,
              padding='valid'))
    model.add(keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(nc, activation='sigmoid'))

    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer='adam')
    return model


class Modeler(object):

    def __init__(self, name, model, nc, cw, sample_num):
        self.name = name
        self.model = model
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
                                           histogram_freq=10,
                                           write_graph=True,
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
            # zoom_range=0.2,
            rotation_range=45,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True)

    def set_x_y(self, x, y):
        self.x_train[self.sample_counter % self.sample_num] = x / (np.power(2, 16) - 1.)
        self.y_train[self.sample_counter % self.sample_num] = y
        self.sample_counter += 1

    def fit_full_datagen(self, epochs, from_epoch=0):
        samples_total = len(self.y_train)
        split = int(samples_total * .2)
        samples = samples_total - split
        x_val, y_val = self.x_train[:split], self.y_train[:split]
        x_train, y_train = self.x_train[split:], self.y_train[split:]
        batch_size = 32
        LOG.info("Training with data generation for model=[%s]" % self.name)
        LOG.info("train samples=[%s], validation samples=[%s], batch size=[%s], epochs=[%s]" % (samples, split, batch_size, epochs))
        steps_per_epochs = samples // batch_size
        self.model.fit_generator(self.datagen.flow(x_train, y_train, batch_size=batch_size),
                                 steps_per_epochs,
                                 epochs=epochs,
                                 verbose=2,
                                 validation_data=(x_val, y_val),
                                 initial_epoch=from_epoch,
                                 callbacks=[self.tb_cb, self.cp_cb])
        """
        self.model.fit(x_train, y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=2,
                       validation_data=(x_val, y_val),
                       initial_epoch=from_epoch,
                       callbacks=[self.tb_cb, self.cp_cb])
        """
        y_train_predict = predict_with_logic(self.model, self.x_train)
        thresh = get_optimal_threshhold(self.y_train, y_train_predict)
        return thresh

    def checkpoint(self):
        LOG.info("Saving model checkpoint for [%s]" % self.name)
        cp_fn = "%s.hdf5" % self.name
        keras.models.save_model(self.model, DH.basepath + "/model/" + cp_fn)
        return


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


def predict_with_logic(m, x):
    Y = m.predict(x)
    for idy, y in enumerate(Y):
        # Chose the most likely single atmospheric condition
        atmos_label_idx = np.argmax(y[:len(ATMOS)])
        y[atmos_label_idx] = 1
        y[:atmos_label_idx] = 0
        y[atmos_label_idx + 1:len(ATMOS)] = 0
        # If it's cloudly, then there are no land-use labels
        if atmos_label_idx == 3:
            y[4:] = 0
        Y[idy] = y
    return Y


def fbeta(true_label, prediction):
    return fbeta_score(true_label, prediction, beta=2, average='samples')


def get_optimal_threshhold(true_label, prediction, iterations=1000):
    best_threshhold = [0.2] * len(LANDUSE)
    for t in range(len(LANDUSE)):
        best_fbeta = 0
        temp_threshhold = [0.2] * len(LANDUSE)
        for i in range(1, iterations + 1):
            temp_value = i / float(iterations)
            temp_threshhold[t] = temp_value
            temp_fbeta = fbeta(true_label[:, len(ATMOS):], prediction[:, len(ATMOS):] > temp_threshhold)
            if temp_fbeta > best_fbeta:
                best_fbeta = temp_fbeta
                best_threshhold[t] = temp_value
    # Always use 0.5 for the atmos threshold since it's already assumed to be 0 or 1 by now
    best_threshhold = [0.5, 0.5, 0.5, 0.5] + best_threshhold
    LOG.info("Using thresholds: %s" % best_threshhold)
    labels_list = ATMOS + LANDUSE
    cm = dict(zip(labels_list + ["all"], [{"tp": 0, "fp": 0, "tn": 0, "fn": 0} for x in range(len(ATMOS + LANDUSE) + 1)]))
    for i, p in enumerate(prediction):
        for j, p_ in enumerate(p):
            if p_ > best_threshhold[j]:
                if true_label[i][j] == 1:
                    cm[labels_list[j]]["tp"] += 1
                    cm["all"]["tp"] += 1
                else:
                    cm[labels_list[j]]["fp"] += 1
                    cm["all"]["fp"] += 1
            else:
                if true_label[i][j] == 1:
                    cm[labels_list[j]]["fn"] += 1
                    cm["all"]["fn"] += 1
                else:
                    cm[labels_list[j]]["tn"] += 1
                    cm["all"]["tn"] += 1
    for k, v in cm.iteritems():
        try:
            precision = v["tp"] / float(v["fp"] + v["tp"])
        except ZeroDivisionError:
            precision = None
        try:
            recall = v["tp"] / float(v["fn"] + v["tp"])
        except ZeroDivisionError:
            recall = None
        try:
            f_beta = 5.0 * ((precision * recall) / ((4 * precision) + recall))
        except ZeroDivisionError:
            f_beta = None
        except TypeError:
            f_beta = None
        LOG.info("%s: [tp=%s, fp=%s, tn=%s, fn=%s]" % (k, v["tp"], v["fp"], v["tn"], v["fn"]))
        LOG.info("%s: [precision=%s, recall=%s, f_beta=%s]" % (k, precision, recall, f_beta))
    return best_threshhold


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


def train(M, imgtyp, epochs, from_epoch=0):
    LOG.info("Starting training run with samples=[%s]" % SAMPLES)
    for X, Y in DH.get_train_iter(imgtyp=imgtyp, h=H, w=W, maxn=SAMPLES):
        M.set_x_y(X, Y.loc[:, ATMOS + LANDUSE].as_matrix()[0])
    thresh = M.fit_full_datagen(epochs=epochs, from_epoch=from_epoch)
    return thresh


def main():
    imgtyp = "tif"
    name = "test-%s" % imgtyp
    submission = "%s.csv" % name
    from_epoch = args.from_epoch
    epochs = from_epoch + 100
    if args.from_saved == "pretrained":
        m = pretrained(len(ATMOS + LANDUSE))
    if not args.from_saved:
        m = multi_label_cnn(len(ATMOS + LANDUSE))
    elif not os.path.isfile(args.from_saved):
        LOG.error("%s is not a file" % args.from_saved)
        return
    else:
        LOG.info("Loading model %s" % args.from_saved)
        m = keras.models.load_model(args.from_saved)
    M = Modeler(name, m, len(ATMOS + LANDUSE), ATMOS_W + LANDUSE_W, SAMPLES)
    if args.all:
        try:
            thresh = train(M, imgtyp, epochs, from_epoch=from_epoch)
        except KeyboardInterrupt:
            LOG.info("Stopping training and checkpointing the model")
            M.checkpoint()
            predictions = M.model.predict(M.x_train)
            thresh = get_optimal_threshhold(M.y_train, predictions)
            write_submission("/output/%s" % submission, M.model, thresh, imgtyp)
            return
        write_submission("/output/%s" % submission, M.model, thresh, imgtyp)
    elif args.train:
        train(M, imgtyp, epochs, from_epoch=from_epoch)
    elif args.test:
        m = keras.models.load_model(args.from_saved)
        name = args.from_saved.split("/")[-1].split(".")[0]
        M = Modeler(name, m, len(ATMOS + LANDUSE), ATMOS_W + LANDUSE_W, SAMPLES)
        for X, Y in DH.get_train_iter(imgtyp=imgtyp, h=H, w=W):
            M.set_x_y(X, Y.loc[:, ATMOS + LANDUSE].as_matrix()[0])
        predictions = predict_with_logic(M.model, M.x_train)
        thresh = get_optimal_threshhold(M.y_train, predictions)
        write_submission("/output/%s" % submission, M.model, thresh, imgtyp)


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
    # sys.stdout = StreamToLogger(LOG)
    # sys.stderr = sys.stdout

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--from-saved', type=str, default=None)
    parser.add_argument('--from-epoch', type=int, default=0)
    args = parser.parse_args()
    main()
