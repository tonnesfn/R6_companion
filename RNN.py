import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.contrib import rnn
import numpy as np
import random
import processScreenshot


class RNN:

    hm_epochs = 100
    n_classes = 65
    batch_size = 128
    n_inputs = 67
    n_steps = 67
    rnn_size = 256

    character_dataset = processScreenshot.CharacterDataset()

    if character_dataset.mode != 'full':
        print('Unsupported mode !')
        exit()

    x = tf.placeholder('float', [None, n_steps, n_inputs])
    y = tf.placeholder('float')

    def recurrent_neural_network(self, x):
        layer = {'weights': tf.Variable(tf.random_normal([self.rnn_size, self.n_classes])),
                 'biases': tf.Variable(tf.random_normal([self.n_classes]))}

        x = tf.transpose(x, [1, 0, 2])
        x = tf.reshape(x, [-1, self.n_inputs])
        x = tf.split(x, self.n_steps, 0)

        lstm_cell = rnn_cell.BasicLSTMCell(self.rnn_size)
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

        output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']

        return output

    def train_neural_network(self):
        prediction = self.recurrent_neural_network(self.x)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=self.y))
        optimizer = tf.train.AdamOptimizer().minimize(cost)

        test_x, test_y = self.character_dataset.get_test_data()
        test_x = np.array(test_x)
        test_y = np.array(test_y)

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            for epoch in range(self.hm_epochs):
                epoch_loss = 0

                i = 0
                while i < int(len(self.character_dataset.characters) * (1 - self.character_dataset.testing_amount)):
                    batch_x, batch_y = self.character_dataset.get_batch(i, self.batch_size);

                    batch_x = np.array(batch_x)
                    batch_y = np.array(batch_y)

                    epoch_x, epoch_y = np.array(batch_x), np.array(batch_y)
                    epoch_x = epoch_x.reshape((self.batch_size, self.n_steps, self.n_inputs))

                    _, c = sess.run([optimizer, cost], feed_dict={self.x: epoch_x, self.y: epoch_y})
                    epoch_loss += c

                    i = i + self.batch_size;

                print('Epoch', epoch, 'completed out of', self.hm_epochs, 'loss:', epoch_loss)

                correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
                print('   Acc:', accuracy.eval({self.x: test_x.reshape((-1, self.n_steps, self.n_inputs)), self.y: test_y}))

    def __init__(self):
        self.character_dataset.load_data_set('dataset')
        random.shuffle(self.character_dataset.characters)
