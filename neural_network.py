import tensorflow as tf


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate, batch_size, training_epochs):
        self.batch_size = batch_size
        self.training_epochs = training_epochs

        self.batch_x = tf.placeholder(tf.float32, [None, input_nodes])
        self.batch_y = tf.placeholder(tf.float32, [None, output_nodes])

        self.weights_input_hidden = tf.Variable(tf.random_normal([input_nodes, hidden_nodes]))
        self.bias_hidden = tf.Variable(tf.random_normal([hidden_nodes]))
        self.input_hidden_layer = tf.nn.sigmoid(tf.add(tf.matmul(self.batch_x, self.weights_input_hidden),
                                                       self.bias_hidden))

        self.weights_hidden_ouput = tf.Variable(tf.random_normal([hidden_nodes, output_nodes]))
        self.bias_output = tf.Variable(tf.random_normal([output_nodes]))

        self.hidden_output_layer = tf.add(tf.matmul(self.input_hidden_layer, self.weights_hidden_ouput),
                                          self.bias_output)

        self.cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.hidden_output_layer,
                                                                                    self.batch_y))
        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.cross_entropy)
        self.session = tf.Session()
        self.model = tf.initialize_all_variables()

    def fit(self, dataset):
        self.session.run(self.model)
        total_batch = int(len(dataset.train_labels) / self.batch_size)

        for epoch in range(self.training_epochs):
            avg_cost = 0
            batch_iterator = dataset.train_batch_iterator(self.batch_size)
            for train, labels in batch_iterator:
                self.session.run(self.train_step, feed_dict={self.batch_x: train, self.batch_y: labels})
                avg_cost += self.session.run(self.cross_entropy,
                                             feed_dict={self.batch_x: train, self.batch_y: labels}) / total_batch
            print("Epoch {:04d} cost= {:.9f}".format(epoch, avg_cost))

        print("Training phase finished")

    def accuracy(self, dataset):
        correct_prediction = tf.equal(tf.argmax(self.batch_y, 1), tf.argmax(tf.nn.sigmoid(self.hidden_output_layer), 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Model accuracy: {0}".format(
            self.session.run(accuracy, feed_dict={self.batch_x: dataset.test, self.batch_y: dataset.test_labels})))

    def __del__(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.session.close()
