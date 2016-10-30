from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random


class Mnist(object):
    def __init__ (self):
        self.mnist = fetch_mldata('MNIST original')
        self.data_graduation()
        self.split_data()

    def show_image(self, image_number):
        image = np.reshape(self.mnist.data[image_number], [28, 28])
        plt.imshow(image, camp='Greys', interpolation='None')
        plt.show()

    def data_graduation(self):
        self.data = np.add(np.dot(np.divide(self.mnist.data, 255), 0.99), 0.01)
        nodes = np.unique(self.mnist.target).size
        labels = []
        for target in self.mnist.target:
            extended_target = np.add(np.zeros(nodes), 0.01)
            extended_target[int(target)] = 0.99
            labels.append(extended_target)
        self.labels = np.asarray(labels, dtype=np.float64)

    def split_data(self):
        self.train, self.test, self.train_labels, self.test_labels = \
            train_test_split(self.data, self.labels, train_size=55/70)

    def train_batch_iterator(self, batch_size):
        #TODO Find more efficient way to do split
        train = list(zip(self.train, self.train_labels))
        random.shuffle(train)
        for i in range(0, len(self.train_labels), batch_size):
            yield zip(*train[i:i + batch_size])