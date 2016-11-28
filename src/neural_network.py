import os
import joblib
import tensorflow as tf
from src.settings import LOG_DIR, OBJECT_DIR


class NeuralNetwork(object):

    def __init__(self, input_nodes: int, hidden_nodes: int, output_nodes: int,
                 learning_rate: float, batch_size: int, training_epochs: int, dropout: float, debug=False):
        self.batch_size = batch_size
        self.training_epochs = training_epochs
        self.debug = debug
        self.dropout = dropout
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        with tf.name_scope('input'):
            self.batch_x = tf.placeholder(tf.float32, [None, input_nodes], name="Batch_x_input")
            self.batch_y = tf.placeholder(tf.float32, [None, output_nodes], name="Batch_y_input")

        self.input_hidden_layer = self.nn_layer(self.batch_x, input_nodes, hidden_nodes,
                                                "Input_Hidden_Layer")

        with tf.name_scope('dropout'):
            self.keep_prob = tf.placeholder(tf.float32)
            tf.scalar_summary('dropout_keep_probability', self.keep_prob)
            self.dropped = tf.nn.dropout(self.input_hidden_layer, self.keep_prob)

        self.hidden_output_layer = self.nn_layer(self.dropped, hidden_nodes,
                                                 output_nodes, 'Hidden_Output_Layer',
                                                 act=tf.identity)

        with tf.name_scope('cross_entropy'):
            self.diff = tf.nn.sigmoid_cross_entropy_with_logits(self.hidden_output_layer,
                                                                self.batch_y)
            with tf.name_scope('total'):
                self.cross_entropy = tf.reduce_mean(self.diff)
            tf.scalar_summary('cross entropy', self.cross_entropy)

        with tf.name_scope('train'):
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cross_entropy)

        with tf.name_scope('test'):
            with tf.name_scope('prediction'):
                self.prediction = tf.argmax(tf.nn.sigmoid(self.hidden_output_layer), 1)

            with tf.name_scope('correct_prediction'):
                self.correct_prediction = tf.equal(tf.argmax(self.batch_y, 1), self.prediction)

            with tf.name_scope('accuracy'):
                self.accuracy_operation = tf.reduce_mean(tf.cast(
                    self.correct_prediction, tf.float32))
            tf.scalar_summary('accuracy', self.accuracy_operation)

        self.session = tf.Session()
        self.model = tf.initialize_all_variables()

        if self.debug:
            self.merged = tf.merge_all_summaries()
            self.train_writer = tf.train.SummaryWriter(os.path.join(LOG_DIR, 'train'),
                                                       self.session.graph)
            self.test_writer = tf.train.SummaryWriter(os.path.join(LOG_DIR, 'test'))

    def predict(self, data, map_function):
        result = self.session.run(self.prediction, feed_dict={self.batch_x: [data], self.keep_prob: 1.0 })[0]
        if callable(map_function):
            result = map_function(result)
        return result

    def fit(self, dataset):
        self.session.run(self.model)
        total_batch = tf.constant(int(len(dataset.train_labels) / self.batch_size),
                                  dtype=tf.float32)
        partial_avg = tf.div(self.cross_entropy, total_batch)
        execution = [self.train_step, partial_avg]
        step = 0
        if self.debug:
            execution.extend([self.merged])

        print("Accuracy without any step= {:.9f}".format(self.accuracy(dataset, -1)))

        for epoch in range(self.training_epochs):
            avg_cost = 0
            batch_iterator = dataset.train_batch_iterator(self.batch_size)
            for train, labels in batch_iterator:
                _, part_avg, *summary = self.session.run(execution,
                                                         feed_dict={self.batch_x: train,
                                                                    self.batch_y: labels,
                                                                    self.keep_prob: self.dropout})
                if self.debug:
                    step += 1
                    self.train_writer.add_summary(summary[0], step)
                avg_cost += part_avg

            print("Epoch {:04d} cost= {:.9f}, accuracy= {:.9f}".format(
                epoch, avg_cost, self.accuracy(dataset, step)))

        print("Training phase finished")

    def accuracy(self, dataset, step=-1):
        execution = [self.accuracy_operation]
        if self.debug:
            execution.append(self.merged)

        accuracy_value, *summary = self.session.run(execution,
                                                    feed_dict={self.batch_x: dataset.test,
                                                               self.batch_y: dataset.test_labels,
                                                               self.keep_prob: 1.0})

        if self.debug:
            self.test_writer.add_summary(summary[0], step)

        return accuracy_value

    def __del__(self):
        if self.debug:
            self.train_writer.close()
            self.test_writer.close()
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if self.debug:
            self.train_writer.close()
            self.test_writer.close()
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

    def save_model(self):
        init_params = {'input_nodes': self.input_nodes, 'hidden_nodes': self.hidden_nodes,
                       'output_nodes': self.output_nodes, 'learning_rate': self.learning_rate,
                       'batch_size': self.batch_size, 'training_epochs': self.training_epochs,
                       'dropout': self.dropout, 'debug': self.debug}
        joblib.dump(init_params, os.path.join(OBJECT_DIR, 'init_params.pickle'))
        tf.train.Saver(tf.all_variables()).save(self.session, os.path.join(OBJECT_DIR, 'model.ckpt'))

    def load_model(self):
        ckpt = tf.train.get_checkpoint_state(OBJECT_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            tf.train.Saver(tf.all_variables()).restore(self.session, ckpt.model_checkpoint_path)

    @staticmethod
    def initialize_weights(shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

    @staticmethod
    def initialize_bias(nodes):
        return tf.Variable(tf.truncated_normal([nodes], stddev=0.1), "Bias Hidden Layer")

    @staticmethod
    def variable_summaries(var, name):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.scalar_summary('mean/' + name, mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.scalar_summary('stddev/' + name, stddev)
            tf.scalar_summary('max/' + name, tf.reduce_max(var))
            tf.scalar_summary('min/' + name, tf.reduce_min(var))
            tf.histogram_summary(name, var)
