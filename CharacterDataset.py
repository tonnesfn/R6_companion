import string
import pickle
import json
import utils
import os
import numpy as np

image_padding_size = [67, 67]


class CharacterDataset:
    testing_amount = 0.2;
    characters = []  # Format: char, data
    training_labels = []
    dataset_dictionary = list(string.ascii_lowercase) + list(string.ascii_uppercase) + \
                         ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '_', '-']

    mode = 'full'  # full / features
    feature_length = 12

    def get_one_hot_encoded(self, character):
        one_hot_encoding = [0] * len(self.dataset_dictionary)  # Space for full dictionary plus padding
        one_hot_encoding[self.dataset_dictionary.index(character)] = 1
        return one_hot_encoding

    def bbox(self, img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        xmin, xmax = np.where(cols)[0][[0, -1]]
        ymin, ymax = np.where(rows)[0][[0, -1]]

        return xmin, xmax, ymin, ymax

    def get_class(self, character):
        return self.dataset_dictionary.index(character)

    def get_char_of_class(self, given_class):
        return self.dataset_dictionary[given_class]

    # Features inspired by http://cns-classes.bu.edu/cn550/Readings/frey-slate-91.pdf
    def get_features(self, image):

        features = [0.0] * self.feature_length

        current_image = image.reshape(image_padding_size[0], image_padding_size[1])

        # Calculate feature 1 and 2: x and y pos of center of bounding box
        bbox = self.bbox(current_image)
        features[0] = np.mean((bbox[0],bbox[1]))
        features[1] = np.mean((bbox[2], bbox[3]))

        # Calculate feature 3 and 4: width and height of bounding box
        features[2] = (bbox[1] - bbox[0]) + 1
        features[3] = (bbox[3] - bbox[2]) + 1

        # Calculate feature 5: Total number of on pixels in image
        features[4] = np.sum(current_image) / np.max(current_image)

        # Calculate feature 6 and 7: Mean horizontal and vertical pos relative to box center
        for y in range(current_image.shape[0]):
            for x in range(current_image.shape[1]):
                if current_image[y][x] == 255:
                    features[5] += x - np.mean((bbox[0], bbox[1]))
                    features[6] += y - np.mean((bbox[2], bbox[3]))

        features[5] = features[5] - features[4]
        features[6] = features[6] - features[4]

        # Calculate feature 8 and 9: Mean squared value of pixel distances as calc in 6
        for y in range(current_image.shape[0]):
            for x in range(current_image.shape[1]):
                if current_image[y][x] == 255:
                    features[7] += np.square(x - np.mean((bbox[0], bbox[1])))
                    features[8] += np.square(y - np.mean((bbox[2], bbox[3])))

        features[7] = features[7] - features[4]
        features[8] = features[8] - features[4]

        # Calculate feature 10: Mean product of horizontal and vertical distances as 6
        features[9] = features[7]*features[8]

        # Calculate feature 11: ?
        features[10] = features[7] * features[6]
        features[11] = features[8] * features[5]

        return features

    def get_batch(self, batch_index, batch_size):
        if len(self.characters) == 0:
            print('Cannot get a batch when no characters have been trained!')

        start = min(batch_index, len(self.characters))

        batch_x = []
        batch_y = []

        for i in range(batch_size):
            if start+i == len(self.characters):
                break;

            if self.mode == 'full':
                batch_x.append(self.characters[start+i][1])
            elif self.mode == 'features':
                batch_x.append(self.get_features(self.characters[start + i][1]))
            else:
                print('Unknown dataset type!')
                exit()

            batch_y.append(self.get_one_hot_encoded(self.characters[start + i][0]))

        return batch_x, batch_y

    # Returns features as X and direct label as Y
    def get_training_data(self):
        train_x = []
        train_y = []

        number_of_samples = int(len(self.characters) * (1-self.testing_amount))-1

        if self.mode == 'features':
            for i in range(number_of_samples):
                train_x.append(self.get_features(self.characters[i][1]))
                train_y.append(self.get_class(self.characters[i][0]))

        return train_x, train_y

    def get_test_data(self, mode='one_hot'):
        start = int(len(self.characters) * (1-self.testing_amount))

        test_x = []
        test_y = []

        for i in range(len(self.characters) - start):

            if self.mode == 'full':
                test_x.append(self.characters[start + i][1])
            elif self.mode == 'features':
                test_x.append(self.get_features(self.characters[start + i][1]))
            else:
                print('Unknown dataset type!')
                exit()

            if mode == 'one_hot':
                test_y.append(self.get_one_hot_encoded(self.characters[start + i][0]))
            elif mode == 'class':
                test_y.append(self.get_class(self.characters[start + i][0]))
            else:
                print('Unknonwn mode!')
                exit()

        return test_x, test_y

    def load_data_set(self, directory):
        if os.path.isfile(directory + '/character_samples.pickle'):
            self.characters = pickle.load(open(directory + '/character_samples.pickle', "rb"))
        else:
            print('No dataset to load - creating new character pickle file')

    def save_data_set(self, directory):
        pickle.dump(self.characters, open(directory + '/character_samples.pickle', "wb"))

        counter_dict = {}
        for character in self.characters:
            if character[0] in counter_dict:
                counter_dict[character[0]] += 1
            else:
                counter_dict[character[0]] = 1

        with open('dataset/character_stats.json', 'w') as outfile:
            json.dump(counter_dict, outfile, indent=4, sort_keys=True)

    def matches(self, character_image_a, character_image_b):
        if len(character_image_a) != len(character_image_b):
            print('Length mismatch!')
            return False

        error = 0

        for i in range(len(character_image_a)):
            error += np.sqrt((float(character_image_a[i]) - float(character_image_b[i])) ** 2.0)

        error = error / len(character_image_a)

        return error

    def classify_character(self, character_image, label=''):

        if (len(self.characters) > 0) and (len(label) == 0):
            # For each character in dataset:
            errors = {}

            for character in self.characters:
                # For each character example in current character:
                all_errors = []
                for i in range(len(character[1])):
                    all_errors.append(self.matches(utils.pil_image_to_array(character_image).flatten(), character[1]))

                errors[character[0]] = min(all_errors)

            if len(self.training_labels) == 0:
                return min(errors, key=errors.get)

        else:
            if len(self.characters) > 0:
                self.characters.append([label, utils.pil_image_to_array(character_image).flatten()])
            else:
                self.characters = [[label, utils.pil_image_to_array(character_image).flatten()]]

            return label

    def classify_sentence(self, character_images, training_label=''):
        classifications = []

        if (len(training_label) != 0) and (len(character_images) != len(training_label)):
            print('Wrong character count when training ' + training_label)
            return None

        for i in range(len(character_images)):
            if len(training_label) == 0:
                classifications.append(self.classify_character(character_images[i]))
            else:
                classifications.append(self.classify_character(character_images[i], training_label[i]))

        return classifications

    def classify_sentences(self, sentence_images, training_labels):

        if (len(self.training_labels) == 0) and (len(self.characters) == 0):
            print('Cannot classify without training data!')
            return None

        classifications = []

        for i in range(len(sentence_images)):
            if len(training_labels) == 0:  # Not training:
                classification = self.classify_sentence(sentence_images[i])
                if classification != None:
                    classifications.append([''.join(classification)])
                else:
                    classifications.append('')
            else:  # Training:
                if len(training_labels[i]) > 0:
                    classification = self.classify_sentence(sentence_images[i], training_labels[i])
                    if classification != None:
                        classifications.append([''.join(classification)])
                    else:
                        classifications.append('')
                else:
                    classifications.append('')

        return classifications
