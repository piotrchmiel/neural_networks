import os
import csv
import src.neural_network as nn
import joblib

from src.settings import OBJECT_DIR


def load_neural_network():
    init_params = joblib.load(os.path.join(OBJECT_DIR, 'init_params.pickle'))
    neural_network = nn.NeuralNetwork(**init_params)
    neural_network.load_model()
    return neural_network


def create_csv(full_file_path, size):
    with open(full_file_path, 'w', newline="") as csvfile:
        fieldnames = ["data{}".format(i) for i in range(size-1)]
        fieldnames.insert(0, "symbol")
        writer = csv.DictWriter(csvfile, fieldnames, delimiter=";")
        writer.writeheader()


def append_row_to_csv(full_file_path, row):
    with open(full_file_path, 'a', newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(row)

def add_new_image(full_file_path, row):
    if not os.path.exists(full_file_path):
        create_csv(full_file_path, len(row))
    append_row_to_csv(full_file_path, row)
