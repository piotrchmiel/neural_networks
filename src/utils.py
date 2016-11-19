import os
import src.neural_network as nn
import joblib

from src.settings import OBJECT_DIR


def load_neural_network():
    init_params = joblib.load(os.path.join(OBJECT_DIR, 'init_params.pickle'))
    neural_network = nn.NeuralNetwork(**init_params)
    neural_network.load_model()
    return neural_network