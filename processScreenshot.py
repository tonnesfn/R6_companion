from PIL import Image
from PIL import ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os.path
import time

blur_amount = 0.5
error_limit = 1.0

pre_spacing = 16
spacing = 46
nick_height = 27

thresholding_limit_black = 120
thresholding_limit_gray = 80
thresholding_limit_white = 90

image_padding_size = [60, 35]


class CharacterDataset:
    characters = {}

    def load_data_set(self, filename):
        if os.path.isfile(filename):
            self.characters = pickle.load(open(filename, "rb"))

    def save_data_set(self, filename):
        pickle.dump(self.characters, open(filename, "wb"))

    def prompt_user_for_class(self, character_image):
        show_image(character_image.filter(ImageFilter.GaussianBlur(blur_amount)))
        print('Unknown character. Please input: >', end='')
        given_character = input()

        if given_character in self.characters:
            self.characters[given_character].append(image_to_array(character_image.filter(ImageFilter.GaussianBlur(blur_amount))).flatten())
        else:
            self.characters[given_character] = [image_to_array(character_image.filter(ImageFilter.GaussianBlur(blur_amount))).flatten()]

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
                    all_errors.append(self.matches(image_to_array(character_image.filter(ImageFilter.GaussianBlur(blur_amount))).flatten(), value[i], key))

                errors[key] = min(all_errors)

            if errors[min(errors, key=errors.get)] < error_limit:
                #print('Matched ' + min(errors, key=errors.get) + ' with error {}'.format(errors[min(errors, key=errors.get)]))
                return min(errors, key=errors.get)


        return self.prompt_user_for_class(character_image)

    def classify_sentence(self, character_images):
        classifications = []

        for i in range(len(character_images)):
            classifications.append(self.classify_character(character_images[i]))

        return classifications

    def classify_sentences(self, sentence_images):
        classifications = []

        for sentence_image in sentence_images:
            classifications.append(''.join(self.classify_sentence(sentence_image)))



        return classifications


def show_image(given_pil_image, figure_name='default_figure'):
    plt.figure(figure_name)
    plt.imshow(given_pil_image, cmap='gray')
    plt.draw()
    plt.pause(0.01)


def image_to_array(img):
    return np.array(img.getdata(),
                    np.uint8).reshape(img.size[1], img.size[0])


def run_custom_filters(given_image):

    # Filter out underscores:
    image_array = image_to_array(given_image)

    last_state = 0
    character_start = 0

    for i in range(len(image_array[-1,:])):
        # Detected transition:
        if image_array[-1, i] != last_state:
            # Moving from character to black
            if last_state == 0:
                character_start = i
            # Moving from black to character:
            else:
                if 14 < i - character_start < 18:
                    #print('len: {}, {} - {}'.format(i-character_start, character_start, i))
                    image_array[:, character_start:character_start+2] = 0
                    image_array[:, i-2:i] = 0

        last_state = image_array[-1, i]

    img_to_return = Image.new("L", given_image.size)
    img_to_return.putdata(image_array.astype(int).flatten())

    return img_to_return

def threshold_image(given_image):
    image_array = (np.asarray(list(given_image.getdata())))
    white_counter = sum((image_array > 100)*1.0) / len(image_array)
    gray_counter = sum((image_array < 150)*1.0) / len(image_array)

    #print('w: {}, g: {}'.format(white_counter, gray_counter), end=' ')

    if gray_counter > 0.99:
        thresholded_list = ((np.asarray(list(given_image.getdata())) > thresholding_limit_gray) * 255)
        #print('gray')
    elif white_counter > 0.8:
        thresholded_list = ((np.asarray(list(given_image.getdata())) < thresholding_limit_white) * 255)
        #print('white')
    else:
        thresholded_list = ((np.asarray(list(given_image.getdata())) > thresholding_limit_black) * 255)
        #print('black')

    ret_image = Image.new('L', given_image.size)
    ret_image.putdata(thresholded_list.astype(int))

    return ret_image


# This function pads an image to a given size and centers the original
def pad_image(given_image):
    padded_image = Image.new('L', (image_padding_size[0], image_padding_size[1]), 0)
    padded_image.paste(given_image, (int((image_padding_size[0]/2) - given_image.size[0]/2), int((image_padding_size[1]/2) - given_image.size[1]/2)))
    return padded_image


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
                characters_images.append(pad_image(given_image.crop((character_start, 0, i, given_image.size[1]))))

        last_state = contains_character[i]

    return characters_images


def process_screenshot(givenImageObject):
    #show_image(givenImageObject, 'whole_image')

    # Crop images into each nick:
    cropped_images = []
    w, h = givenImageObject.size
    for i in range(5):
        # crop: left, upper, right, lower
        cropped_images.append(threshold_image(givenImageObject.crop((0, pre_spacing + (spacing*i)+(i*nick_height), w, pre_spacing + (spacing*i)+((i+1)*nick_height)))))

    for i in range(len(cropped_images)):
        cropped_images[i] = run_custom_filters(cropped_images[i])

    character_lists = []

    for i in range(len(cropped_images)):
        character_lists.append(split_nick_image(cropped_images[i]))

    return character_lists

if __name__ == "__main__":
    character_dataset = CharacterDataset()
    character_dataset.load_data_set('dataset/character_dataset.pickle')

    im1 = Image.open("screenshot_examples/bottom_2017_07_15_113641.jpg").convert('L')
    sentence_images = process_screenshot(im1)
    nicks = CharacterDataset.classify_sentences(character_dataset, sentence_images)

    character_dataset.save_data_set('dataset/character_dataset.pickle')

    #im2 = Image.open("bottom_2017_07_13_234146.jpg").convert('L')
    #process_screenshot(im2)