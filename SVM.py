import numpy as np
import processScreenshot
import random
from sklearn import preprocessing, cross_validation, neighbors, svm

class SVM:

    character_dataset = processScreenshot.CharacterDataset()

    def train_svm(self):

        x_train, y_train = self.character_dataset.get_training_data()
        x_test, y_test = self.character_dataset.get_test_data('class')

        clf = svm.SVC()
        clf.fit(np.array(x_train),  np.array(y_train))

        accuracy = clf.score(np.array(x_test), np.array(y_test))
        print(accuracy)

    def __init__(self):
        self.character_dataset.load_data_set('dataset')
        random.shuffle(self.character_dataset.characters)
