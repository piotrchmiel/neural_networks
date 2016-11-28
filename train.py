import os
import joblib
from src.dataset import Mnist, Uji
from src.neural_network import NeuralNetwork
from src.settings import OBJECT_DIR


def main():
    dataset = Mnist()
    #dataset = Uji()
    joblib.dump(dataset, os.path.join(OBJECT_DIR, "Mnist"))
    #joblib.dump(dataset, os.path.join(OBJECT_DIR, "Uji"))

    with NeuralNetwork(input_nodes=dataset.feature_number, hidden_nodes=250, output_nodes=dataset.label_number,
                       learning_rate=0.01, batch_size=100, training_epochs=30, dropout=0.6,
                       debug=False) as nn:
        nn.fit(dataset)
        nn.save_model()


if __name__ == '__main__':
    main()
