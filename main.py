from src.dataset import Mnist, Uji
from src.neural_network import NeuralNetwork


def main():
    # mnist = Mnist()
    uji = Uji()
    with NeuralNetwork(input_nodes=784, hidden_nodes=100, output_nodes=10,
                       learning_rate=0.01, batch_size=100, training_epochs=10, debug=True) as nn:
        nn.fit(uji)

if __name__ == '__main__':
    main()
