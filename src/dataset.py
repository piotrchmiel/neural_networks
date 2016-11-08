import abc
import os
import random
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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
        return

    def show_image(self, image_number):
        image = np.reshape(self.dataset.data[image_number], [28, 28])
        plt.imshow(image, camp='Greys', interpolation='None')
        plt.show()

    @abc.abstractmethod
    def data_graduation(self):
        return

    def split_data(self):
        self.train, self.test, self.train_labels, self.test_labels = \
            train_test_split(self.data, self.labels, train_size=55/70)

    def train_batch_iterator(self, batch_size):
        #TODO Find more efficient way to do split
        train = list(zip(self.train, self.train_labels))
        random.shuffle(train)
        for i in range(0, len(self.train_labels), batch_size):
            yield zip(*train[i:i + batch_size])


class Mnist(HandwrittenDataset):

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
        self.labels = np.asarray(labels, dtype=np.float64)


class Uji(HandwrittenDataset):

    def __init__(self):
        self.encoder = LabelEncoder()
        super(Uji, self).__init__()

    def load_data(self):
        self.dataset = UJIData()

    def data_graduation(self):
        self.data = self.dataset.data
        self.labels = self.encoder.fit_transform(self.dataset.target)


HandwrittenDataset.register(Mnist)
HandwrittenDataset.register(Uji)


class UJIData(object):

    def __init__(self, filenames=None):
        if filenames is None:
            datasets_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "datasets")
            filenames = [os.path.join(datasets_dir, "UJI1.csv"), os.path.join(datasets_dir, "UJI2.csv")]
        self.target = []
        self.data = None
        self.load_data(filenames)

    def load_data(self, filenames):
        for file in filenames:
            data = pd.read_csv(file, sep=';', dtype={'symbol': object})
            labels = data['symbol']
            del data['symbol']
            data = np.array(data)
            if self.data is not None:
                self.data = np.concatenate((self.data, data))
            else:
                self.data = data
            self.target += list(labels)
