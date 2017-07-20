import tensorflow as tf
import numpy as np
import processScreenshot
import random


class ANN:
    n_nodes_hl1 = 10000
    n_nodes_hl2 = 10000
    n_nodes_hl3 = 10000

    n_classes = 65
    batch_size = 50

    character_dataset = processScreenshot.CharacterDataset()

    if character_dataset.mode == 'features':
        input_length = character_dataset.feature_length
    elif character_dataset.mode == 'full':
        input_length = 4489
    else:
        print('Unknown mode !')
        exit()

    x = tf.placeholder('float', [None, input_length])
    y = tf.placeholder('float')

    def neural_network_model(self, data):
        hidden_1_layer = {'weights': tf.Variable(tf.random_normal([self.input_length, self.n_nodes_hl1])),
                          'biases': tf.Variable(tf.random_normal([self.n_nodes_hl1]))}
        hidden_2_layer = {'weights': tf.Variable(tf.random_normal([self.n_nodes_hl1, self.n_nodes_hl2])),
                          'biases': tf.Variable(tf.random_normal([self.n_nodes_hl2]))}
        hidden_3_layer = {'weights': tf.Variable(tf.random_normal([self.n_nodes_hl2, self.n_nodes_hl3])),
                          'biases': tf.Variable(tf.random_normal([self.n_nodes_hl3]))}
        output_layer   = {'weights': tf.Variable(tf.random_normal([self.n_nodes_hl3, self.n_classes])),
                          'biases': tf.Variable(tf.random_normal([self.n_classes]))}

        l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
        l1 = tf.nn.relu(l1)

        l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
        l2 = tf.nn.relu(l2)

        l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
        l3 = tf.nn.relu(l3)

        output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

        return output

    def train_neural_network(self):
        prediction = self.neural_network_model(self.x)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=self.y))
        optimizer = tf.train.AdamOptimizer().minimize(cost)

        hm_epochs = 100

        test_x, test_y = self.character_dataset.get_test_data()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(hm_epochs):
                epoch_loss = 0

                i = 0
                while i < int(len(self.character_dataset.characters) * (1-self.character_dataset.testing_amount)):
                    batch_x, batch_y = self.character_dataset.get_batch(i, self.batch_size);

                    batch_x = np.array(batch_x)
                    batch_y = np.array(batch_y)

                    _, c = sess.run([optimizer, cost], feed_dict={self.x: batch_x, self.y: batch_y})
                    epoch_loss += c

                    i = i + self.batch_size;

                print('Epoch', epoch+1, 'completed out of', hm_epochs, 'loss:', epoch_loss)

                correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

                print('   Acc:', accuracy.eval({self.x: test_x, self.y: test_y}))

    def __init__(self):
        self.character_dataset.load_data_set('dataset')
        random.shuffle(self.character_dataset.characters)
