from src.dataset import Mnist
from src.neural_network import NeuralNetwork


def main():
    mnist = Mnist()
    with NeuralNetwork(input_nodes=784, hidden_nodes=100, output_nodes=10,
                       learning_rate=0.01, batch_size=100, training_epochs=10, debug=True) as nn:
        nn.fit(mnist)

if __name__ == '__main__':
    main()
