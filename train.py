import os
import joblib
from src.dataset import Mnist, Uji
from src.neural_network import NeuralNetwork
from src.settings import OBJECT_DIR


def main():
    # mnist = Mnist()
    uji = Uji()
    joblib.dump(uji, os.path.join(OBJECT_DIR, "Uji"))

    with NeuralNetwork(input_nodes=uji.feature_number, hidden_nodes=100, output_nodes=uji.label_number,
                       learning_rate=0.01, batch_size=100, training_epochs=10, dropout=0.6,
                       debug=False) as nn:
        nn.fit(uji)
        nn.save_model()


if __name__ == '__main__':
    main()
