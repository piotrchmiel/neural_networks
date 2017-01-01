import os.path
from src.dataset import Uji
from src.neural_network import NeuralNetwork
from src.settings import BASE_DIR
import tensorflow as tf


def main():
    dataset = Uji()
    OPTIMIZERS = [tf.train.RMSPropOptimizer, tf.train.AdamOptimizer, tf.train.GradientDescentOptimizer,
                  tf.train.AdagradOptimizer, tf.train.AdadeltaOptimizer]
    LEARNING_RATES = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    TRAINING_EPOCHS = [10, 20, 50, 100, 200, 300, 400, 500, 700, 1000]
    RESULTS = []

    for optimizer in OPTIMIZERS:
        print(optimizer)
        for learning_rate in LEARNING_RATES:
            print("Learning rate: {}".format(learning_rate))
            with NeuralNetwork(input_nodes=dataset.feature_number, hidden_nodes=200, output_nodes=dataset.label_number,
                               learning_rate=learning_rate, batch_size=100,training_epochs=TRAINING_EPOCHS[-1],
                               dropout=0.6, optimizer=optimizer, debug=False) as nn:
                accuracy_results = nn.fit(dataset, TRAINING_EPOCHS)
                for accuracy, training_epoch in zip(accuracy_results, TRAINING_EPOCHS):
                    RESULTS.append(tuple([accuracy, optimizer, learning_rate, training_epoch]))

    RESULTS.sort(key=lambda row: row[0][0], reverse=True)

    with open(os.path.join(BASE_DIR, 'parametr_stats_by_accuracy.txt'), 'w') as fh:
        for snapshot_results, optimizer, learning_rate, training_epochs in RESULTS:
            fh.write("Accuracy {0} Cost {1} Optimizer {2} Learning Rate {3}, Training Epoches: {4}\n".format(
                snapshot_results[0], snapshot_results[1], optimizer, learning_rate, training_epochs))

    RESULTS.sort(key=lambda row: row[0][1])

    with open(os.path.join(BASE_DIR, 'parametr_stats_by_cost.txt'), 'w') as fh:
        for snapshot_results, optimizer, learning_rate, training_epochs in RESULTS:
            fh.write("Accuracy {0} Cost {1} Optimizer {2} Learning Rate {3}, Training Epoches: {4}\n".format(
                snapshot_results[0], snapshot_results[1], optimizer, learning_rate, training_epochs))

if __name__ == '__main__':
    main()