from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import CharacterDataset
import cv2
import ScreenshotCapture
import utils

blur_amount = 0.5

underscore_position = 47

thresholding_limit_black = 165  # Higher number is a stricter thresholding
thresholding_limit_gray = 95    # Higher number is a stricter thresholding
thresholding_limit_white = 88   # Lower number is a stricter thresholding

image_padding_size = [67, 67]


def get_sentence_images_from_screenshot(given_image):
    name_width = 465
    left_offset = 42
    lines = []

    if given_image.size == (2560, 1440):
        left, tops = 415, [439, 512, 585, 659, 732]
        line_height = 67
    else:
        print('Unknown screenshot size {}{}!'.format(given_image.size[0],given_image.size[1]))
        exit()

    for i in range(5):
        lines.append(given_image.crop((left+left_offset, tops[i], left+name_width, tops[i]+line_height)))

    if given_image.size == (2560, 1440):
        left, tops = 415, [929, 1003, 1076, 1149, 1223]
        line_height = 67
    else:
        print('Unknown screenshot size {}{}!'.format(given_image.size[0], given_image.size[1]))
        exit()

    for i in range(5):
        lines.append(given_image.crop((left+left_offset, tops[i], left+name_width, tops[i]+line_height)))

    return lines


def get_letter_images_from_screenshot(given_image):
    sentences = get_sentence_images_from_screenshot(given_image)
    return process_screenshot(sentences)

def run_custom_filters(given_image):

    # Filter out underscores:
    image_array = utils.pil_image_to_array(given_image)

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
                    image_array[underscore_position-1:underscore_position+2, character_start-2:character_start+3] = 0
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

    if gray_counter > 0.99:
        thresholded_list = ((np.asarray(list(given_image.getdata())) > thresholding_limit_gray) * 255)
    elif white_counter > 0.65:
        thresholded_list = ((np.asarray(list(given_image.getdata())) < thresholding_limit_white) * 255)
    else:
        thresholded_list = ((np.asarray(list(given_image.getdata())) > thresholding_limit_black) * 255)

    ret_image = Image.new('L', given_image.size)
    ret_image.putdata(thresholded_list.astype(int))

    if gray_counter > 0.99:
        ret_image = erode_image(ret_image, 1)
    elif white_counter > 0.65:
        ret_image = erode_image(ret_image, 1)
        image_array = utils.pil_image_to_array(ret_image)
        image_array[0:5, :] = 0
        image_array[underscore_position + 1:, :] = 0
        ret_image = utils.array_to_pil_image(image_array, ret_image)
    else:
        ret_image = erode_image(ret_image, 1)

    return ret_image


# This function pads an image to a given size and centers the original
def pad_image(given_image):
    padded_image = Image.new('L', (image_padding_size[0], image_padding_size[1]), 0)
    padded_image.paste(given_image, (int((image_padding_size[0]/2) - given_image.size[0]/2), int((image_padding_size[1]/2) - given_image.size[1]/2)))
    return padded_image


def tightly_crop(given_image):

    image_array = utils.pil_image_to_array(given_image)
    new_array = np.sum(given_image, axis=1)

    min_index = min(loc for loc, val in enumerate(new_array) if val > 0)
    max_index = max(loc for loc, val in enumerate(new_array) if val > 0)

    ret_image = Image.new('L', (given_image.size[0], (max_index-min_index)+1))
    ret_image.putdata(image_array[min_index:max_index+1, :].astype(int).flatten())

    return ret_image


def split_nick_image(given_image):

    contains_character = (np.sum(utils.pil_image_to_array(given_image), axis=0) > 0)

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


def erode_image(given_image, kernel_size):
    image_array = utils.pil_image_to_array(given_image)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    image_array = cv2.erode(image_array, kernel, iterations=1)
    img_to_return = Image.new("L", given_image.size)
    img_to_return.putdata(image_array.astype(int).flatten())

    return img_to_return


def dilate_image(given_image, kernel_size):
    image_array = utils.pil_image_to_array(given_image)
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
    character_dataset = CharacterDataset.CharacterDataset()

    character_dataset.load_data_set('dataset')
    character_dataset.training_labels = training_labels

    letters = process_screenshot(given_sentence_images)
    nicks = CharacterDataset.classify_sentences(character_dataset, letters, training_labels)

    if len(training_labels) > 0:
        character_dataset.save_data_set()

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

