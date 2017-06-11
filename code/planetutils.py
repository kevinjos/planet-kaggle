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
        self.test_jpg = "test-jpg-v2"
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

    def _get_iter(self, samp, imgtyp="tif", h=256, w=256, maxn=None):
        expected_samp = ("train", "test")
        expected_imgtyp = ("jpg", "tif")
        path = self.path + "/"
        if samp not in expected_samp or imgtyp not in expected_imgtyp:
            raise NameError(samp, "%s found but expected string in %s or %s" % (samp, expected_samp, expected_imgtyp))
        elif samp == "train" and imgtyp == "tif":
            path += self.train_tif
        elif samp == "train" and imgtyp == "jpg":
            path += self.train_jpg
        elif samp == "test" and imgtyp == "tif":
            path = self.basepath + "/data/v2/" + self.test_tif
        elif samp == "test" and imgtyp == "jpg":
            path = self.basepath + "/data/v2/" + self.test_jpg
        files = os.listdir(path)
        random.shuffle(files)
        for fn in files[:maxn]:
            name = fn.split(".")[0]
            img = cv2.imread(path + "/" + fn, -1)
            img = cv2.resize(img, (h, w))
            yield((name, img))


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

    def get_train_iter(self, imgtyp="tif", h=256, w=256, maxn=None):
        train_iter = self._get_iter("train", imgtyp=imgtyp, h=h, w=w, maxn=maxn)
        for name, X in train_iter:
            Y = self.train_labels.loc[self.train_labels["name"] == name]
            yield (X, Y)

    def get_test_iter(self, imgtyp="tif", h=256, w=256, maxn=None):
        test_iter = self._get_iter("test", imgtyp=imgtyp, h=h, w=w, maxn=maxn)
        return test_iter
