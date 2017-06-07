import pandas as pd
import numpy as np
import cv2
import os
import random


class FileHandler(object):
    def __init__(self, basepath="/Users/kjs/repos/planet"):
        # Directories
        self.basepath = basepath
        self.path = basepath + "/input"
        self.train_tif = "train-tif-v2"
        self.train_jpg = "train-jpg"
        self.validation_jpg = "validation-jpg"
        self.test_tif = "test-tif-v2"
        self.test_jpg = "test-jpg"
        # Files
        self.train_labels_csv = "train_v2.csv"
        # Data
        self._train_labels = None
        self._train_cached = False
        self.train_cache = {}

    def _set_train_labels(self):
        if self._train_labels is not None:
            return
        with open(self.path + "/" + self.train_labels_csv, "r") as fp:
            res = {}
            fp.readline()  # Skip the header
            for l in fp:
                k, v = l.split(",")
                v = v.strip().split(" ")
                res[k] = v
        self._train_labels = res
        return

    def set_train_cache(self, h, w):
        path = self.path + "/" + self.train_tif
        files = os.listdir(path)
        random.shuffle(files)
        self._set_train_cache(path, files, h, w)

    def _set_train_cache(self, path, files, h, w):
        for fn in files:
            name = fn.split(".")[0]
            img = cv2.imread(path + "/" + fn, -1)
            img = cv2.resize(img, (h, w))
            self.train_cache[name] = img
        self._train_cached = True
        return

    def _get_tif_iter(self, samp, h=256, w=256, do_cache=False):
        expected = ("train", "test")
        path = self.path + "/"
        if samp not in expected:
            raise NameError(samp, "%s found but expected string in %s" % (samp, expected))
        elif samp == "train":
            path += self.train_tif
        elif samp == "test":
            path = self.basepath + "/data/v2/" + self.test_tif
        files = os.listdir(path)
        random.shuffle(files)
        if do_cache and not self._train_cached:
            self._set_train_cache(path, files, h, w)
        if not do_cache:
            for fn in files:
                name = fn.split(".")[0]
                img = cv2.imread(path + "/" + fn, -1)
                img = cv2.resize(img, (h, w))
                yield((name, img))
        else:
            for name, img in self.train_cache.iteritems():
                yield((name, img))

    def get_tif(self, name, h=256, w=256):
        path = self.path + "/" + self.train_tif + "/" + name
        img = cv2.imread(path, -1)
        img = cv2.resize(img, (h, w))
        return img

    def get_train_jpg_path(self):
        return self.path + "/" + self.train_jpg

    def get_validation_jpg_path(self):
        return self.path + "/" + self.validation_jpg


class DataHandler(FileHandler):
    def __init__(self, **kwargs):
        super(DataHandler, self).__init__(**kwargs)
        self.train_labels = None

    def set_train_labels(self):
        if self.train_labels is not None:
            return
        self._set_train_labels()
        pd_encoded = {"name": []}
        for k, v in self._train_labels.iteritems():
            for t in v:
                if t not in pd_encoded:
                    pd_encoded[t] = []
        for k, v in self._train_labels.iteritems():
            for pd_k in pd_encoded.keys():
                if pd_k == "name":
                    pd_encoded["name"].append(k)
                    continue
                pd_encoded[pd_k].append(pd_k in v)
        pd_encoded = pd.DataFrame(pd_encoded)
        self.train_labels = pd_encoded

    def get_train_iter(self, h=256, w=256):
        train_iter = self._get_tif_iter("train", h, w)
        for name, d in train_iter:
            Y = self.train_labels.loc[self.train_labels["name"] == name]
            X = d
            yield (X, Y)

    def get_test_iter(self, h=256, w=256):
        test_iter = self._get_tif_iter("test", h, w)
        return test_iter

    def get_manual_train_iter(self, h=256, w=256):
        train_iter = self._get_tif_iter("train", h, w)
        for name, d in train_iter:
            Y = self.train_labels.loc[self.train_labels["name"] == name]
            X = d
            yield (name, X, Y)
