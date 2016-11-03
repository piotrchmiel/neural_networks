import tensorflow as tf
from src.settings import LOG_DIR


class NeuralNetwork(object):

    def __init__(self, input_nodes: int, hidden_nodes: int, output_nodes: int,
                 learning_rate: float, batch_size: int, training_epochs: int, debug=False):
        self.batch_size = batch_size
        self.training_epochs = training_epochs
        self.debug = debug

        with tf.name_scope('input'):
            self.batch_x = tf.placeholder(tf.float32, [None, input_nodes], name="Batch_x_input")
            self.batch_y = tf.placeholder(tf.float32, [None, output_nodes], name="Batch_y_input")

        self.input_hidden_layer = self.nn_layer(self.batch_x, input_nodes, hidden_nodes, "Input_Hidden_Layer")
        self.hidden_output_layer = self.nn_layer(self.input_hidden_layer, hidden_nodes, output_nodes,
                                                 'Hidden_Output_Layer', act=tf.identity)

        with tf.name_scope('cross_entropy'):
            self.diff = tf.nn.sigmoid_cross_entropy_with_logits(self.hidden_output_layer,self.batch_y)
            with tf.name_scope('total'):
                self.cross_entropy = tf.reduce_mean(self.diff)
            tf.scalar_summary('cross entropy', self.cross_entropy)

        with tf.name_scope('train'):
            self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.cross_entropy)

        self.session = tf.Session()
        self.model = tf.initialize_all_variables()

        if self.debug:
            self.merged = tf.merge_all_summaries()
            self.train_writer = tf.train.SummaryWriter(LOG_DIR, self.session.graph)

    def fit(self, dataset):
        self.session.run(self.model)
        total_batch = int(len(dataset.train_labels) / self.batch_size)
        step = 1

        for epoch in range(self.training_epochs):
            avg_cost = 0
            batch_iterator = dataset.train_batch_iterator(self.batch_size)
            for train, labels in batch_iterator:
                if self.debug:
                    summary, _ = self.session.run([self.merged, self.train_step], feed_dict={self.batch_x: train,
                                                                                             self.batch_y: labels})
                    self.train_writer.add_summary(summary, step)
                    step += 1
                else:
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
        if self.debug:
            self.train_writer.close()
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if self.debug:
            self.train_writer.close()
        self.session.close()

    def nn_layer(self, input_tensor, input_dim, output_dim, layer_name, act=tf.nn.sigmoid):
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                weights = self.initialize_weights([input_dim, output_dim])
                if self.debug:
                    self.variable_summaries(weights, layer_name + '/weights')
            with tf.name_scope('biases'):
                biases = self.initialize_bias(output_dim)
                if self.debug:
                    self.variable_summaries(biases, layer_name + '/biases')
            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.add(tf.matmul(input_tensor, weights), biases)
                tf.histogram_summary(layer_name + '/pre_activations', preactivate)
            activations = act(preactivate, 'activation')
            tf.histogram_summary(layer_name + '/activations', activations)
            return activations

    def initialize_weights(self, shape):
        return tf.Variable(tf.random_normal(shape))

    def initialize_bias(self, nodes):
        return tf.Variable(tf.random_normal([nodes]), "Bias Hidden Layer")

    def variable_summaries(self, var, name):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.scalar_summary('mean/' + name, mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.scalar_summary('stddev/' + name, stddev)
            tf.scalar_summary('max/' + name, tf.reduce_max(var))
            tf.scalar_summary('min/' + name, tf.reduce_min(var))
            tf.histogram_summary(name, var)
