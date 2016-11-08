from src.dataset import Mnist, Uji
from src.neural_network import NeuralNetwork


def main():
    # mnist = Mnist()
    uji = Uji()
    uji.show_image(3)
    with NeuralNetwork(input_nodes=uji.feature_number, hidden_nodes=100, output_nodes=uji.label_number,
                       learning_rate=0.01, batch_size=100, training_epochs=100, debug=False) as nn:
        nn.fit(uji)

if __name__ == '__main__':
    main()
