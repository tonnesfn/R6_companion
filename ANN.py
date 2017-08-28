import tensorflow as tf
import numpy as np
import processScreenshot
import CharacterDataset
import random


class ANN:
    n_nodes_hl1 = 1000
    n_nodes_hl2 = 1000
    n_nodes_hl3 = 1000

    n_classes = 65
    batch_size = 128

    character_dataset = CharacterDataset.CharacterDataset()

    if character_dataset.mode == 'features':
        input_length = character_dataset.feature_length
    elif character_dataset.mode == 'full':
        input_length = 4489
    else:
        print('Unknown mode !')
        exit()

    sess = None
    prediction = None

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
        saver = tf.train.Saver(max_to_keep=1000)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=self.y))
        optimizer = tf.train.AdamOptimizer().minimize(cost)

        hm_epochs = 100

        #test_x, test_y = self.character_dataset.get_test_data()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(hm_epochs):
                epoch_loss = 0

                i = 0
                #while i < int(len(self.character_dataset.characters) * (1-self.character_dataset.testing_amount)):
                while i < int(len(self.character_dataset.characters)):
                    batch_x, batch_y = self.character_dataset.get_batch(i, self.batch_size);

                    batch_x = np.array(batch_x)
                    batch_y = np.array(batch_y)

                    _, c = sess.run([optimizer, cost], feed_dict={self.x: batch_x, self.y: batch_y})
                    epoch_loss += c

                    i = i + self.batch_size

                print('Epoch', epoch+1, 'completed out of', hm_epochs, 'loss:', epoch_loss)

                #correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y, 1))
                #accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

                #acc = accuracy.eval({self.x: test_x, self.y: test_y})

                #print('Acc: {0:.2f}'.format(acc))

                saver.save(sess, '/models/ANN/model-{}-{}.ckpt'.format(epoch, epoch_loss))

    def get_prediction(self, given_images):
        if self.sess == None:
            self.prediction = self.neural_network_model(self.x)

            saver = tf.train.Saver(max_to_keep=1000)
            self.sess = tf.InteractiveSession()
            self.sess.run(tf.global_variables_initializer())
            saver.restore(self.sess, "models/ANN/model-83-311392.40625.ckpt")

        result = []

        for image in given_images:
            res = (self.sess.run(tf.argmax(self.prediction.eval(feed_dict={self.x: [image]}), 1)))
            result.append(self.character_dataset.get_char_of_class(res[0]))

        return result

    def __init__(self):
        self.character_dataset.load_data_set('dataset')
        random.shuffle(self.character_dataset.characters)


if __name__ == "__main__":
    character_dataset = processScreenshot.CharacterDataset()
    character_dataset.load_data_set('dataset')

    ann = ANN()

    correct = 0
    wrong = 0

    for character in character_dataset.characters:
        inp = []
        inp.append(character[1])
        pred = ann.get_prediction(inp)

        if pred[0].lower() != character[0].lower():
            wrong += 1
            #processScreenshot.array_to_image(character[1], 67, 67).show()
            print('Wrong: actual: ' + character[0] + ', predicted: ' + pred[0])
        else:
            correct += 1

    print('Total {}/{} wrong)'.format(wrong, correct))
