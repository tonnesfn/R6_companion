from PIL import Image
from PIL import ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os.path
import cv2
import time
import json
import ScreenshotCapture

blur_amount = 0.5
error_limit = 0.1

pre_spacing = 14
spacing = 43
underscore_position = 38
nick_height = 31

thresholding_limit_black = 120  # Higher number is a stricter thresholding
thresholding_limit_gray = 110  # Higher number is a stricter thresholding
thresholding_limit_white = 80  # Lower number is a stricter thresholding

image_padding_size = [60, 35]


class CharacterDataset:
    characters = {}
    currently_training = False

    def load_data_set(self, filename):
        if os.path.isfile(filename):
            self.characters = pickle.load(open(filename, "rb"))

    def save_data_set(self, directory):
        pickle.dump(self.characters, open(directory + '/character_samples.pickle', "wb"))

        counterDict = {}
        for key, value in self.characters.items():
            counterDict[key] = len(value)

        with open('dataset/character_stats.json', 'w') as outfile:
            json.dump(counterDict, outfile, indent=4, sort_keys=True)

    def prompt_user_for_class(self, character_image):
        #show_image(character_image.filter(ImageFilter.GaussianBlur(blur_amount)))
        show_image(character_image)
        print('> ', end='')

        while True:
            given_character = input()
            if len(given_character) == 1:
                break

        if given_character in self.characters:
            #self.characters[given_character].append(image_to_array(character_image.filter(ImageFilter.GaussianBlur(blur_amount))).flatten())
            self.characters[given_character].append(image_to_array(character_image).flatten())
        else:
            #self.characters[given_character] = [image_to_array(character_image.filter(ImageFilter.GaussianBlur(blur_amount))).flatten()]
            self.characters[given_character] = [image_to_array(character_image).flatten()]

        return given_character

    def matches(self, character_image_a, character_image_b, character_print):
        if len(character_image_a) != len(character_image_b):
            print('Length mismatch!')
            return False

        error = 0

        for i in range(len(character_image_a)):
            error += np.sqrt((float(character_image_a[i]) - float(character_image_b[i])) ** 2.0)

        error = error / len(character_image_a)

        #print('Error for ' + character_print + ': {}'.format(error))

        return error

    def classify_character(self, character_image):

        if len(self.characters) > 0:
            # For each character in dataset:
            errors = {}

            for key, value in self.characters.items():
                # For each character example in current character:
                all_errors = []
                for i in range(len(value)):
                    #all_errors.append(self.matches(image_to_array(character_image.filter(ImageFilter.GaussianBlur(blur_amount))).flatten(), value[i], key))
                    all_errors.append(self.matches(image_to_array(character_image).flatten(), value[i], key))

                errors[key] = min(all_errors)

            if self.currently_training == False:
                return min(errors, key=errors.get)

        return self.prompt_user_for_class(character_image)

    def classify_sentence(self, character_images):
        classifications = []

        for i in range(len(character_images)):
            classifications.append(self.classify_character(character_images[i]))

        return classifications

    def classify_sentences(self, sentence_images):

        if (self.currently_training == False) and (len(self.characters) == 0):
            print('Cannot classify without training data!')
            return None

        classifications = []

        for sentence_image in sentence_images:
            classifications.append(''.join(self.classify_sentence(sentence_image)))

        return classifications


def show_image(given_pil_image, figure_name='default_figure'):
    plt.figure(figure_name)
    plt.imshow(given_pil_image, cmap='gray')
    plt.draw()
    plt.pause(0.01)


def array_to_image(given_array, original_image):
    img_to_return = Image.new("L", original_image.size)
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
        if max(image_array[underscore_position-1:underscore_position+1, i]) != last_state:
            # Moving from character to black
            if last_state == 0:
                character_start = i
            # Moving from black to character:
            else:
                # If a single underscore:
                if 10 < i - character_start < 18:

                    underscore_height = image_array[:, int(((i - character_start) / 2) + character_start)] == 255

                    min_index = min(loc for loc, val in enumerate(underscore_height) if val == True)
                    max_index = max(loc for loc, val in enumerate(underscore_height) if val == True)

                    #print('len: {}, {} - {}'.format(i-character_start, character_start, i))
                    image_array[min_index:max_index+1, character_start-2:character_start+2] = 0
                    image_array[min_index:max_index+1, i-2:i+2] = 0

        last_state = max(image_array[underscore_position-1:underscore_position+1, i])

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

    character_lists = []

    for i in range(len(given_images)):
        character_lists.append(split_nick_image(given_images[i]))

    return character_lists


def get_nicks(given_sentence_images, currently_training=False):
    character_dataset = CharacterDataset()

    character_dataset.load_data_set('dataset/character_samples.pickle')
    character_dataset.currently_training = currently_training

    letters = process_screenshot(given_sentence_images)
    nicks = CharacterDataset.classify_sentences(character_dataset, letters)

    character_dataset.save_data_set('dataset')

    return nicks

if __name__ == "__main__":

    screenshot_capture = ScreenshotCapture.ScreenshotCapture()
    screenshot_example = Image.open('screenshot_examples/T-2017_07_16_133135.jpg').convert('L')
    screenshot_capture.set_screenshot(screenshot_example)

    print(get_nicks(screenshot_capture.get_top_names(), True))
