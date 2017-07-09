import tensorflow.contrib.keras as keras
from tensorflow.contrib.keras import backend as K
import tensorflow as tf
import numpy as np
from sklearn.metrics import fbeta_score

import logging
import os
import argparse

from planetutils import DataHandler, mkdir
from planetmodels import multi_label_cnn, pretrained, unet


ATMOS = ['clear', 'partly_cloudy', 'haze', 'cloudy']
LANDUSE = ['primary', 'agriculture', 'road', 'water', 'cultivation', 'habitation', 'bare_ground',
           'artisinal_mine', 'blooming', 'blow_down', 'selective_logging', 'slash_burn', 'conventional_mine']
LABELS = ATMOS + LANDUSE
LABELS_N = len(LABELS)
IMGTYP = 'tif'
H, W, CHANS = 128, 128, 4 if IMGTYP == 'tif' else 3
IMG_SHAPE = (W, H, CHANS)


class Modeler(object):

    def __init__(self, name, model, output_basepath):
        self.name = name
        self.model = model
        self.basepath = output_basepath
        # Image remixer
        self.datagen = self.train_datagen()
        # Keras modeling callbacks
        self.tb_cb = self.tensorboard_cb()
        self.cp_cb = self.checkpoint_cb()
        self.el_cb = self.epochlog_cb()
        # Data
        self.x_train = None
        self.x_val = None
        self.y_train = None
        self.y_val = None
        self.x_test = None
        self.x_mean = None
        self.x_std = None

    # Setup tensorboard callbacks
    def tensorboard_cb(self):
        graphdir = self.basepath + '/graph/' + self.name + '/'
        mkdir(graphdir)
        return keras.callbacks.TensorBoard(log_dir=graphdir,
                                           histogram_freq=25,
                                           write_graph=True,
                                           write_images=False)

    # Setup model checkpoint callbacks
    def checkpoint_cb(self):
        modeldir = self.basepath + '/model/' + self.name + '/'
        mkdir(modeldir)
        cp_fn = '{epoch:03d}-{val_loss:.5f}.hdf5'
        return keras.callbacks.ModelCheckpoint(modeldir + cp_fn, monitor='val_loss',
                                               save_best_only=True,
                                               save_weights_only=False,
                                               mode='auto',
                                               period=1)

    def epochlog_cb(self):
        return EpochLogger(logger=LOG)

    def train_datagen(self):
        return keras.preprocessing.image.ImageDataGenerator(
            shear_range=0.2,
            # zoom_range=0.2,
            rotation_range=45,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True)

    def set_validation_split(self, validation_fraction):
        samples_total = len(self.y_train)
        split = int(samples_total * validation_fraction)
        self.x_val = self.x_train[:split]
        self.y_val = self.y_train[:split]
        self.x_train = self.x_train[split:]
        self.y_train = self.y_train[split:]
        LOG.info("Training model=[%s] with train samples=[%s] and validation samples=[%s]" % (self.name, self.x_train.shape[0], self.x_val.shape[0]))

    def set_mean_and_std(self, mean=None, std=None):
        self.x_mean = mean
        self.x_std = std
        if mean is None:
            self.x_mean = np.mean(self.x_train, axis=(0, 1, 2))
        if std is None:
            self.x_std = np.std(self.x_train, axis=(0, 1, 2))
        LOG.info("By-channel mean=[%s] and std=[%s]" % (self.x_mean, self.x_std))

    def set_train_norm(self):
        self.x_train -= self.x_mean
        self.x_train /= self.x_std
        self.x_val -= self.x_mean
        self.x_val /= self.x_std

    def set_test_norm(self):
        self.x_test -= self.x_mean
        self.x_test /= self.x_std

    def fit_full_datagen(self, epochs=1, from_epoch=0, batch_size=128):
        steps_per_epochs = self.x_train.shape[0] // batch_size
        self.model.fit_generator(self.datagen.flow(self.x_train, self.y_train, batch_size=batch_size),
                                 steps_per_epochs,
                                 epochs=epochs,
                                 verbose=0,
                                 validation_data=(self.x_val, self.y_val),
                                 initial_epoch=from_epoch,
                                 callbacks=[self.tb_cb, self.cp_cb, self.el_cb])

    def set_threshholds(self, thresh=None):
        self.thresh = thresh
        if thresh is None:
            predictions = self.predict_val()
            self.thresh = get_optimal_threshhold(self.y_val, predictions)

    def predict_test(self):
        Y = self.model.predict(self.x_test)
        np.apply_along_axis(atmos_prediction, Y)
        return Y > self.thresh

    def predict_val(self):
        Y = self.model.predict(self.x_val)
        np.apply_along_axis(atmos_prediction, Y)
        return Y


def atmos_prediction(y):
    most_likely = np.argmax(y[:len(ATMOS)])
    y[most_likely] = 1.0
    y[most_likely + 1:len(ATMOS)] = 0.0
    y[:most_likely] = 0.0


def get_optimal_threshhold(true_label, prediction, iterations=1000):
    best_threshhold = [0.0 for x in range(len(ATMOS) + len(LANDUSE))]
    for t in range(len(ATMOS) + len(LANDUSE)):
        best_fbeta = 0
        for i in range(1, iterations + 1):
            temp_value = i / float(iterations)
            temp_fbeta = fbeta_score(true_label[:, t], prediction[:, t] > temp_value, beta=2, average='binary')
            if temp_fbeta > best_fbeta:
                best_fbeta = temp_fbeta
                best_threshhold[t] = temp_value
    LOG.info('Using thresholds: %s' % best_threshhold)
    labels_list = LABELS
    cm = dict(zip(labels_list + ['all'], [{'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0} for x in range(len(ATMOS + LANDUSE) + 1)]))
    for i, p in enumerate(prediction):
        for j, p_ in enumerate(p):
            if p_ > best_threshhold[j]:
                if true_label[i][j] == 1:
                    cm[labels_list[j]]['tp'] += 1
                    cm['all']['tp'] += 1
                else:
                    cm[labels_list[j]]['fp'] += 1
                    cm['all']['fp'] += 1
            else:
                if true_label[i][j] == 1:
                    cm[labels_list[j]]['fn'] += 1
                    cm['all']['fn'] += 1
                else:
                    cm[labels_list[j]]['tn'] += 1
                    cm['all']['tn'] += 1
    for k, v in cm.iteritems():
        try:
            precision = v['tp'] / float(v['fp'] + v['tp'])
        except ZeroDivisionError:
            precision = None
        try:
            recall = v['tp'] / float(v['fn'] + v['tp'])
        except ZeroDivisionError:
            recall = None
        try:
            f_beta = 5.0 * ((precision * recall) / ((4 * precision) + recall))
        except ZeroDivisionError:
            f_beta = None
        except TypeError:
            f_beta = None
        LOG.info('%s: [tp=%s, fp=%s, tn=%s, fn=%s]' % (k, v['tp'], v['fp'], v['tn'], v['fn']))
        LOG.info('%s: [precision=%s, recall=%s, f_beta=%s]' % (k, precision, recall, f_beta))
    return best_threshhold


def write_predictions(Y, names, output_basepath, model_name):
    with open(output_basepath + "/" + model_name + '.csv', 'w') as fp:
        fp.write('image_name,tags\n')
        for y, n in zip(Y, names):
            row = n + ','
            for i, p in enumerate(y):
                if p:
                    row += LABELS[i]
                    row += ' '
            row = row.strip()
            row += '\n'
            fp.write(row)


class EpochLogger(keras.callbacks.Callback):
    def __init__(self, logger):
        self.log = logger

    def on_epoch_end(self, epoch, logs):
        self.log.info('Epoch %s: loss=[%s], val_loss=[%s]' % (epoch, logs['loss'], logs['val_loss']))


def main():
    maxn = None
    DH = DataHandler(input_basepath=input_basepath)
    # Load in a model either from a file or from scratch
    if args.from_saved is not None:
        name = args.from_saved.split("/")[-1]
        LOG.info('Loading model=[%s]' % name)
        m = keras.models.load_model(args.from_saved)
    else:
        name = '{model_name}-{img_typ}'.format(model_name='foo', img_typ=IMGTYP)
        LOG.info('Initializing model=[%s]' % name)
        m = multi_label_cnn(LABELS_N, H, W, CHANS)
    M = Modeler(name, m, output_basepath)
    if args.train:
        train_iter = DH.get_train_iter(imgtyp=IMGTYP, h=H, w=W, maxn=maxn)
        M.x_train = np.empty(shape=(maxn, W, H, CHANS), dtype='float32')
        M.y_train = np.empty(shape=(maxn, LABELS_N), dtype='bool')
        for i, (x, y) in enumerate(train_iter):
            M.x_train[i] = x
            M.y_train[i] = y.loc[:, LABELS].as_matrix()[0]
        M.set_validation_split()
        M.set_mean_and_std()
        M.set_train_norm()
        M.fit_full_datagen(epochs=args.from_epoch + 100, from_epoch=args.from_epoch, batch_size=128)
        M.set_threshholds()
    elif args.test:
        test_iter = DH.get_test_iter(imgtyp=IMGTYP, h=H, w=W, maxn=maxn)
        M.x_test = np.empty(shape=(maxn, W, H, CHANS), dtype='float32')
        names = np.empty(shape=(maxn), dtype='S10')
        for i, (name, x) in enumerate(test_iter):
            M.x_test[i] = x
            names[i] = name
        M.set_mean_and_std(mean=np.array(args.mean, dtype='float32'), std=np.array(args.std, dtype='float32'))
        M.set_test_norm()
        M.set_treshholds(thresh=np.array(args.thresh, dtype='float32'))
        Y = M.predict_test()
        write_predictions(Y, names, output_basepath, name)


if __name__ == '__main__':
    input_basepath, output_basepath = "/Users/kjs/repos/planet", "/Users/kjs/repos/planet"
    LOG = logging.getLogger(__name__)
    LOG.setLevel(logging.INFO)
    logdir = output_basepath + '/log/'
    mkdir(logdir)
    logfile = logdir + "planet-kaggle.log"
    handler = logging.FileHandler(logfile)
    formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    LOG.addHandler(handler)

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--from-saved', type=str, default=None)
    parser.add_argument('--from-epoch', type=int, default=0)
    parser.add_argument('--thresh', type=float, nargs='+')
    parser.add_argument('--mean', type=float, nargs='+')
    parser.add_argument('--std', type=float, nargs='+')
    args = parser.parse_args()
    try:
        main()
    except Exception as e:
        LOG.error("Exception: %s" % e, exc_info=True)
