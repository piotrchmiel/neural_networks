import abc
import os
import random
import zipfile
import collections

from functools import partial

from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split

from src.settings import BASE_DIR
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class HandwrittenDataset(metaclass=abc.ABCMeta):

    def __init__(self):
        self.train, self.test, self.train_labels, self.test_labels = None, None, None, None
        self.load_data()
        self.data_graduation()
        self.split_data()

    @abc.abstractmethod
    def load_data(self):
        raise NotImplementedError

    def show_image(self, image):
        image = np.reshape(image, [28, 28])
        plt.imshow(image, cmap='Greys', interpolation='None')
        plt.show()

    @abc.abstractmethod
    def data_graduation(self):
        raise NotImplementedError

    def split_data(self):
        self.train, self.test, self.train_labels, self.test_labels = \
            train_test_split(self.data, self.labels, train_size=0.8)

    def train_batch_iterator(self, batch_size):
        #TODO Find more efficient way to do split
        train = list(zip(self.train, self.train_labels))
        random.shuffle(train)
        for i in range(0, len(self.train_labels), batch_size):
            yield zip(*train[i:i + batch_size])

    @property
    def feature_number(self):
        return len(self.data[0])

    @property
    def label_number(self):
        return len(set(self.dataset.target))


class Mnist(HandwrittenDataset):

    def __init__(self):
        super().__init__()

    def load_data(self):
        self.dataset = fetch_mldata('MNIST original')

    def data_graduation(self):
        self.data = np.add(np.dot(np.divide(self.dataset.data, 255), 0.99), 0.01)
        nodes = np.unique(self.dataset.target).size
        labels = []
        for target in self.dataset.target:
            extended_target = np.add(np.zeros(nodes), 0.01)
            extended_target[int(target)] = 0.99
            labels.append(extended_target)
        self.labels = np.asarray(labels, dtype=np.float32)


class Uji(HandwrittenDataset):

    def __init__(self):
        super().__init__()

    def load_data(self):
        self.dataset = UJIData()

    def data_graduation(self):
        self.data = self.dataset.data
        self.label_encoder()

    def _init_encoded_labels(self, unique_classes):
        return np.add(np.zeros(len(unique_classes)), 0.01)

    def label_encoder(self):
        self.unique_classes = sorted(tuple(set(self.dataset.target)))
        self.encoded_labels = collections.defaultdict(partial(self._init_encoded_labels, self.unique_classes))
        self.labels = list()
        for i, label in enumerate(self.unique_classes):
            self.encoded_labels[label][i] = 0.99

        for target in self.dataset.target:
            self.labels.append(self.encoded_labels[target])

        self.labels = np.asarray(self.labels, dtype=np.float32)

    def map_result(self, index):
        return self.unique_classes[index]


class UJIData(object):

    def __init__(self, filenames=None):
        if filenames is None:
            datasets_dir = os.path.join(BASE_DIR, "datasets")
            self.filenames = [os.path.join(datasets_dir, "UJI1.csv"), os.path.join(datasets_dir, "UJI2.csv")]

        try:
            self.verify_filenames_exist()
        except OSError as e:
            compressed = self.find_related_compressed_files(e, '.zip')
            self._extract_files(compressed)

        self.target = []
        self.data = None
        self.load_data()

    def load_data(self):
        for file in self.filenames:
            data = pd.read_csv(file, sep=';', dtype={'symbol': object})
            labels = data['symbol']
            del data['symbol']
            data = np.array(data)
            if self.data is not None:
                self.data = np.concatenate((self.data, data))
            else:
                self.data = data
            self.target += list(labels)

    def verify_filenames_exist(self):
        for path in self.filenames:
            if not os.path.exists(path):
                raise OSError(path + " doesn't exist")

    def find_related_compressed_files(self, error, extension):
        compressed_files = []
        for path in self.filenames:
            new_path = "".join([os.path.splitext(path)[0], extension])
            if os.path.exists(new_path):
                compressed_files.append(new_path)
            else:
                raise OSError(new_path + " doesn't exist") from error
        return compressed_files

    def _extract_files(self, filenames):
        print("Extracting files...")
        for path in filenames:
            print("Compressed file {0}".format(path))
            with zipfile.ZipFile(path, 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(path))
        print("Done.")
