import os.path
from src.dataset import Uji
from src.neural_network import NeuralNetwork
from src.settings import BASE_DIR
import tensorflow as tf


def main():
    dataset = Uji()
    OPTIMIZERS = [tf.train.AdamOptimizer, tf.train.GradientDescentOptimizer,
                  tf.train.AdagradOptimizer, tf.train.AdadeltaOptimizer, tf.train.MomentumOptimizer,
                  tf.train.RMSPropOptimizer]
    LEARNING_RATES = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    TRAINING_EPOCHS = [10, 20, 50, 100, 200, 300, 400, 500, 700, 1000]
    RESULTS = []
    for optimizer in OPTIMIZERS:
        print(optimizer)
        for learning_rate in LEARNING_RATES:
            print("Learning rate: {}".format(learning_rate))
            for training_epochs in TRAINING_EPOCHS:
                print("Training epochs: {}".format(training_epochs))
                with NeuralNetwork(input_nodes=dataset.feature_number, hidden_nodes=200,
                                   output_nodes=dataset.label_number, learning_rate=learning_rate, batch_size=100,
                                   training_epochs=training_epochs, dropout=0.6, optimizer=optimizer,
                                   debug=False) as nn:
                    nn.fit(dataset)
                    RESULTS.append(tuple([nn.accuracy(dataset), optimizer, learning_rate, training_epochs]))

    RESULTS.sort(key=lambda row: row[0], reverse=True)

    with open(os.path.join(BASE_DIR, 'parametr_stats.txt'), 'w') as fh:
        for accuracy, optimizer, learning_rate, training_epochs in RESULTS:
            fh.write("Accuracy {0} Optimizer {1} Learning Rate {2}, Training Epoches: {3}\n".format(
                accuracy, optimizer, learning_rate, training_epochs))

if __name__ == '__main__':
    main()