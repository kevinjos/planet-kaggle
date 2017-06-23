import tensorflow.contrib.keras as keras
from tensorflow.contrib.keras import backend as K
import tensorflow as tf
import numpy as np
from sklearn.metrics import fbeta_score

import logging
import os
import argparse

from planetutils import DataHandler
from planetmodels import multi_label_cnn, pretrained, unet


ATMOS = ['clear',
         'partly_cloudy',
         'haze', 'cloudy']
ATMOS_W = [1, 2, 4, 4]
LANDUSE = ['primary', 'agriculture', 'road', 'water',
           'cultivation', 'habitation',
           'bare_ground', 'artisinal_mine', 'blooming', 'blow_down', 'selective_logging', 'slash_burn', 'conventional_mine']
LANDUSE_W = [1, 2, 2, 2, 4, 4, 8, 8, 8, 8, 8, 8, 8]


IMGTYP = 'tif'

H, W, CHANS = 128, 128, 4 if IMGTYP == 'tif' else 3
IMG_SHAPE = (W, H, CHANS)

DH = DataHandler(basepath='/mnt/planet-kaggle')

DH.set_train_labels()

SAMPLES = 100
if SAMPLES is None:
    SAMPLES = DH.train_labels.shape[0]


class Modeler(object):

    def __init__(self, name, model, nc, sample_num, cw=None):
        self.name = name
        self.model = model
        self.class_weight = cw
        self.sample_num = sample_num
        self.datagen = self.train_datagen()
        self.tb_cb = self.tensorboard_cb()
        self.cp_cb = self.checkpoint_cb()
        self.el_cb = self.epochlog_cb()
        self.x_train = np.empty(shape=(self.sample_num, W, H, CHANS), dtype='float32')
        self.y_train = np.empty(shape=(self.sample_num, nc), dtype='bool')
        self.sample_counter = 0

    def __repr__(self):
        return self.name

    # Setup tensorboard callbacks
    def tensorboard_cb(self, basepath=DH.basepath):
        graphdir = basepath + '/graph/' + self.name + '/'
        mkdir(graphdir)
        return keras.callbacks.TensorBoard(log_dir=graphdir,
                                           histogram_freq=10,
                                           write_graph=True,
                                           write_images=False)

    # Setup model checkpoint callbacks
    def checkpoint_cb(self, basepath=DH.basepath):
        modeldir = basepath + '/model/' + self.name + '/'
        mkdir(modeldir)
        cp_fn = '{epoch:03d}-{val_loss:.5f}.hdf5'
        return keras.callbacks.ModelCheckpoint(modeldir + cp_fn, monitor='val_loss',
                                               save_best_only=True,
                                               save_weights_only=False,
                                               mode='auto',
                                               period=2)

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

    def set_x_y(self, x, y, x_transform):
        self.x_train[self.sample_counter % self.sample_num] = x_transform(x)
        self.y_train[self.sample_counter % self.sample_num] = y
        self.sample_counter += 1

    def fit_full_datagen(self, epochs, from_epoch=0):
        samples_total = len(self.y_train)
        split = int(samples_total * .2)
        samples = samples_total - split
        x_val, y_val = self.x_train[:split], self.y_train[:split]
        x_train, y_train = self.x_train[split:], self.y_train[split:]
        x_train_mean_c = np.mean(x_train, axis=(0, 1, 2))
        x_train_std_c = np.std(x_train, axis=(0, 1, 2))
        self.x_train -= x_train_mean_c
        self.x_train /= x_train_std_c
        batch_size = 128
        LOG.info('Training with data generation for model=[%s]' % self.name)
        LOG.info('train samples=[%s], validation samples=[%s], batch size=[%s], epochs=[%s]' % (samples, split, batch_size, epochs))
        LOG.info('train data channel means=%s & stdev=%s' % (x_train_mean_c, x_train_std_c))
        steps_per_epochs = samples // batch_size
        self.model.fit_generator(self.datagen.flow(x_train, y_train, batch_size=batch_size),
                                 steps_per_epochs,
                                 epochs=epochs,
                                 verbose=0,
                                 validation_data=(x_val, y_val),
                                 initial_epoch=from_epoch,
                                 # callbacks=[self.el_cb])
                                 callbacks=[self.tb_cb, self.cp_cb, self.el_cb])
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
        LOG.info('Saving model checkpoint for [%s]' % self.name)
        cp_fn = '%s.hdf5' % self.name
        keras.models.save_model(self.model, DH.basepath + '/model/' + cp_fn)
        return


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
    LOG.info('Using thresholds: %s' % best_threshhold)
    labels_list = ATMOS + LANDUSE
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


def prediction(M, X, thresh):
    model_prediction = predict_with_logic(M, np.array([X]))
    model_prediction = model_prediction[0]
    labels = ATMOS + LANDUSE
    for i, elem in enumerate(model_prediction):
        model_prediction[i] = model_prediction[i] > thresh[i]
    result = []
    for i, label in enumerate(labels):
        if model_prediction[i]:
            result.append(label)
    return ' '.join(result)


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


def write_submission(outputpath, m, thresh, x_transform):
    maxn = None if SAMPLES > 10000 else 100
    with open(DH.basepath + outputpath, 'w') as fp:
        fp.write('image_name,tags\n')
        for name, X in DH.get_test_iter(imgtyp=IMGTYP, h=H, w=W, maxn=maxn):
            X = x_transform(X)
            p = prediction(m, X, thresh)
            fp.write('%s,%s\n' % (name, p))


def mkdir(d):
    if not os.path.exists(d):
        os.mkdir(d)


def train(M, epochs, x_transform, from_epoch=0):
    LOG.info('Starting training run with samples=[%s]' % SAMPLES)
    for X, Y in DH.get_train_iter(imgtyp=IMGTYP, h=H, w=W, maxn=SAMPLES):
        M.set_x_y(X, Y.loc[:, ATMOS + LANDUSE].as_matrix()[0], x_transform)
    thresh = M.fit_full_datagen(epochs=epochs, from_epoch=from_epoch)
    return thresh


def main():
    x_transform = lambda x: x
    name = 'vgg16-he-wi-%s' % IMGTYP
    submission = '%s.csv' % name
    if args.train:
        from_epoch = args.from_epoch
        epochs = from_epoch + 100
        if not args.from_saved:
            m = multi_label_cnn(len(ATMOS + LANDUSE), H, W, CHANS)
        else:
            LOG.info('Loading model %s' % args.from_saved)
            m = keras.models.load_model(args.from_saved)
        M = Modeler(name, m, len(ATMOS + LANDUSE), SAMPLES)
        train(M, epochs, x_transform, from_epoch=from_epoch)
    elif args.test:
        m = keras.models.load_model(args.from_saved)
        name = args.from_saved.split('/')[-1].split('.')[0]
        M = Modeler(name, m, len(ATMOS + LANDUSE), SAMPLES)
        for X, Y in DH.get_train_iter(imgtyp=IMGTYP, h=H, w=W):
            M.set_x_y(X, Y.loc[:, ATMOS + LANDUSE].as_matrix()[0], x_transform)
        predictions = predict_with_logic(M.model, M.x_train)
        thresh = get_optimal_threshhold(M.y_train, predictions)
        write_submission('/output/%s' % submission, M.model, thresh, x_transform)


class EpochLogger(keras.callbacks.Callback):
    def __init__(self, logger):
        self.log = logger
    def on_epoch_end(self, epoch, logs):
        t = self.model.optimizer.iterations + 1
        self.log.info("Epoch %s: loss=[%s], val_loss=[%s]" % (epoch, logs['loss'], logs['val_loss']))


if __name__ == '__main__':
    LOG = logging.getLogger(__name__)
    LOG.setLevel(logging.INFO)
    logfile = DH.basepath + '/log/' + 'planet-kaggle-cnn-v5.log'
    handler = logging.FileHandler(logfile)
    formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    LOG.addHandler(handler)

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--from-saved', type=str, default=None)
    parser.add_argument('--from-epoch', type=int, default=0)
    args = parser.parse_args()
    try:
        main()
    except Exception as e:
        LOG.error("Exception: %s" % e, exc_info=True)
