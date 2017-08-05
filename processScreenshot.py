from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os.path
import cv2
import json
import ScreenshotCapture
import string

blur_amount = 0.5

underscore_position = 47

thresholding_limit_black = 140  # Higher number is a stricter thresholding
thresholding_limit_gray = 95    # Higher number is a stricter thresholding
thresholding_limit_white = 85   # Lower number is a stricter thresholding

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
                    all_errors.append(self.matches(image_to_array(character_image).flatten(), character[1]))

                errors[character[0]] = min(all_errors)

            if len(self.training_labels) == 0:
                return min(errors, key=errors.get)

        else:
            if len(self.characters) > 0:
                self.characters.append([label, image_to_array(character_image).flatten()])
            else:
                self.characters = [[label, image_to_array(character_image).flatten()]]

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


def show_image(given_pil_image, figure_name='default_figure'):
    plt.figure(figure_name)
    plt.imshow(given_pil_image, cmap='gray')
    plt.draw()
    plt.pause(0.01)


def array_to_image(given_array, par1, par2=None):

    if par2==None:
        img_to_return = Image.new("L", par1.size)
    else:
        img_to_return = Image.new("L", (par1, par2))

    img_to_return.putdata(given_array.astype(int).flatten())
    return img_to_return


def image_to_array(img):
    return np.array(img.getdata(),
                    np.uint8).reshape(img.size[1], img.size[0])


def run_custom_filters(given_image):

    # Filter out underscores:
    image_array = image_to_array(given_image)

    last_state = 0
    character_start = 0

    for i in range(len(image_array[underscore_position,:])):
        # Detected transition:
        if max(image_array[underscore_position-1:underscore_position+2, i]) != last_state:
            # Moving from character to black
            if last_state == 0:
                character_start = i
            # Moving from black to character:
            else:
                # If a single underscore:
                if 10 < i - character_start < 18:

                    #print('len: {}, {} - {}'.format(i-character_start, character_start, i))
                    image_array[underscore_position-1:underscore_position+2, character_start-2:character_start+2] = 0
                    image_array[underscore_position-1:underscore_position+2, i-2:i+2] = 0

        last_state = max(image_array[underscore_position-1:underscore_position+2, i])

    img_to_return = Image.new("L", given_image.size)
    img_to_return.putdata(image_array.astype(int).flatten())

    return img_to_return


def save_pil_image(given_image, filename):
    given_image.save(filename, "JPEG", quality=100)


def threshold_image(given_image):
    image_array = (np.asarray(list(given_image.getdata())))
    white_counter = sum((image_array > 100)*1.0) / len(image_array)
    gray_counter = sum((image_array < 150)*1.0) / len(image_array)

    #print('w: {}, g: {}'.format(white_counter, gray_counter), end=' ')

    if gray_counter > 0.99:
        thresholded_list = ((np.asarray(list(given_image.getdata())) > thresholding_limit_gray) * 255)
        #print('gray')
    elif white_counter > 0.65:
        thresholded_list = ((np.asarray(list(given_image.getdata())) < thresholding_limit_white) * 255)
        #print('white')
    else:
        thresholded_list = ((np.asarray(list(given_image.getdata())) > thresholding_limit_black) * 255)
        #print('black')

    ret_image = Image.new('L', given_image.size)
    ret_image.putdata(thresholded_list.astype(int))

    if gray_counter > 0.99:
        ret_image = erode_image(ret_image, 1)
    elif white_counter > 0.65:
        ret_image = erode_image(ret_image, 1)
        image_array = image_to_array(ret_image)
        image_array[0:5, :] = 0
        image_array[underscore_position + 1:, :] = 0
        ret_image = array_to_image(image_array, ret_image)
    else:
        ret_image = erode_image(ret_image, 1)

    return ret_image


# This function pads an image to a given size and centers the original
def pad_image(given_image):
    padded_image = Image.new('L', (image_padding_size[0], image_padding_size[1]), 0)
    padded_image.paste(given_image, (int((image_padding_size[0]/2) - given_image.size[0]/2), int((image_padding_size[1]/2) - given_image.size[1]/2)))
    return padded_image


def tightly_crop(given_image):

    image_array = image_to_array(given_image)
    new_array = np.sum(given_image, axis=1)

    min_index = min(loc for loc, val in enumerate(new_array) if val > 0)
    max_index = max(loc for loc, val in enumerate(new_array) if val > 0)

    ret_image = Image.new('L', (given_image.size[0], (max_index-min_index)+1))
    ret_image.putdata(image_array[min_index:max_index+1, :].astype(int).flatten())

    return ret_image

def split_nick_image(given_image):
    #given_image.show()

    contains_character = (np.sum(image_to_array(given_image), axis=0) > 0)

    last_state = False
    character_start = 0
    characters_images = []

    for i in range(len(contains_character)):
        # Detected transition:
        if contains_character[i] != last_state:
            # Moving from character to black
            if last_state == False:
                character_start = i
            # Moving from black to character:
            else:
                #characters_images.append(pad_image(tightly_crop(given_image.crop((character_start, 0, i, given_image.size[1])))))
                characters_images.append(pad_image(given_image.crop((character_start, 0, i, given_image.size[1]))))

        last_state = contains_character[i]

    return characters_images


def erode_image(given_image, kernel_size):
    image_array = image_to_array(given_image)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    image_array = cv2.erode(image_array, kernel, iterations=1)
    img_to_return = Image.new("L", given_image.size)
    img_to_return.putdata(image_array.astype(int).flatten())

    return img_to_return

def dilate_image(given_image, kernel_size):
    image_array = image_to_array(given_image)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    image_array = cv2.dilate(image_array, kernel, iterations=1)
    img_to_return = Image.new("L", given_image.size)
    img_to_return.putdata(image_array.astype(int).flatten())

    return img_to_return

def process_screenshot(given_images):
    for i in range(len(given_images)):
        given_images[i] = threshold_image(given_images[i])
        given_images[i] = run_custom_filters(given_images[i])
        #given_images[i].show()

    character_lists = []

    for i in range(len(given_images)):
        character_lists.append(split_nick_image(given_images[i]))

    return character_lists


def get_letters(images1, images2):
    letters_list = process_screenshot(images1) + process_screenshot(images2)

    sentences = []

    for sentence in letters_list:
        letters = []
        for letter in sentence:
            letters.append(np.asarray(letter).flatten())
        sentences.append(letters)

    return sentences


def get_nicks(given_sentence_images, training_labels=[]):
    character_dataset = CharacterDataset()

    character_dataset.load_data_set('dataset')
    character_dataset.training_labels = training_labels

    letters = process_screenshot(given_sentence_images)
    nicks = CharacterDataset.classify_sentences(character_dataset, letters, training_labels)

    if len(training_labels) > 0:
        character_dataset.save_data_set('dataset')

    return nicks

if __name__ == "__main__":

    screenshot_capture = ScreenshotCapture.ScreenshotCapture()
    screenshot_example = Image.open('screenshot_examples/screenshot_2017_07_17_185104.jpg').convert('L')
    screenshot_capture.set_screenshot(screenshot_example)

    top_names, bottom_names = screenshot_capture.get_names()

    top_team = get_nicks(top_names)
    bottom_team = get_nicks(bottom_names)

    print(top_team)
    print(bottom_team)

