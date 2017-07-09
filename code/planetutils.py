import pandas as pd
import numpy as np
import cv2
import os
import random


def mkdir(d):
    if not os.path.exists(d):
        os.mkdir(d)


class FileHandler(object):
    def __init__(self, input_basepath="/Users/kjs/repos/planet"):
        # Directories
        self.basepath = input_basepath
        self.path = self.basepath + "/input"
        self.train_tif = "train-tif"
        self.test_jpg = "test-jpg"
        self.train_jpg = "train-jpg"
        self.test_tif = "test-tif"
        # Files
        self.train_labels_csv = "train_v2.csv"
        # Stats
        self.train_n = 40479
        self.test_n = 61191

    def _get_train_labels(self):
        with open(self.path + "/" + self.train_labels_csv, "r") as fp:
            res = {}
            fp.readline()  # Skip the header
            for l in fp:
                k, v = l.split(",")
                v = v.strip().split(" ")
                res[k] = v
        return res

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
        train_labels_raw = self._get_train_labels()
        pd_encoded = {"name": []}
        for k, v in train_labels_raw.iteritems():
            for t in v:
                if t not in pd_encoded:
                    pd_encoded[t] = []
        for k, v in train_labels_raw.iteritems():
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
