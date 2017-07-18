import processScreenshot
from PIL import Image
import json
import os.path
import ScreenshotCapture
import lazysiege

import RNN

sample_directory = 'screenshot_examples/'
labels_json_file_name = 'labels.json'

def label_dataset(given_filename):
    # Open existing json
    if os.path.isfile(sample_directory + labels_json_file_name):
        with open(sample_directory + labels_json_file_name, 'r') as infile:
            labels = json.load(infile)
    else:
        labels = []

    overwrite_index = -1
    # Check if the file has already been labeled:
    for i in range(len(labels)):
        if labels[i]['filename'] == given_filename:
            print('Filename exists, overwrite? y/n')
            if input() != 'y':
                print('Exiting program')
                exit()
            else:
                overwrite_index = i

    # Open image file
    current_image = Image.open(sample_directory + given_filename)
    current_image.show()

    # Get input nicks and verify names:
    names = []

    print('Please write names:')

    for i in range(10):
        while True:
            nick_input = input('')

            if len(nick_input) == 0:
                print('Added empty slot')
                names.append('')
                break
            else:

                returned_nick = lazysiege.lookup_player(nick_input)['player']['username']

                if returned_nick != 'notfound':
                    print('Added ' + returned_nick)
                    names.append(returned_nick)
                    break
                else:
                    print(nick_input + ' is not a valid name! Please try again.')

    # Check for valid names
    if overwrite_index == -1:
        labels.append(dict(filename=given_filename, names=names))
    else:
        labels[overwrite_index] = dict(filename=given_filename, names=names)

    with open(sample_directory + labels_json_file_name, 'w') as outfile:
        json.dump(labels, outfile, indent=4, sort_keys=True)

def train(given_model):

    if given_model == 'RNN':
        rnn = RNN.RNN()
        rnn.train_neural_network()

    else:
        print('Unknown model!')

def generate_dataset(given_file_name):
    already_trained_on = []

    if os.path.isfile('dataset/readImages.json') == True:
        with open('dataset/readImages.json') as f:
            already_trained_on = json.load(f)

    if sample_directory + given_file_name in already_trained_on:
        print('Already generated samples from ' + given_file_name)
    else:
        # Open existing json
        if os.path.isfile(sample_directory + labels_json_file_name):
            with open(sample_directory + labels_json_file_name, 'r') as infile:
                labels = json.load(infile)
        else:
            labels = []

        image_labels = []
        # Check if the file has already been labeled:
        for i in range(len(labels)):
            if labels[i]['filename'] == given_file_name:
                image_labels = labels[i]['names']

        if len(image_labels) == 0:
            print('Current image has not been labeled!')
            exit()

        screenshot_capture = ScreenshotCapture.ScreenshotCapture()
        screenshot_example = Image.open(sample_directory + given_file_name).convert('L')
        screenshot_capture.set_screenshot(screenshot_example)

        top_names, bottom_names = screenshot_capture.get_names()

        processScreenshot.get_nicks(top_names, image_labels[:5])
        processScreenshot.get_nicks(bottom_names, image_labels[5:])

        already_trained_on.append(sample_directory + given_file_name)
        with open('dataset/readImages.json', 'w') as outfile:
            json.dump(already_trained_on, outfile, indent=4, sort_keys=True)


#sample_directory = 'screenshot_examples/'
#current_file = 'screenshot_2017_07_17_141006.jpg'

if __name__ == "__main__":

    sample_files = os.listdir(sample_directory)
    sample_files.remove(labels_json_file_name)

    label_file_contents = []
    if os.path.isfile(sample_directory + labels_json_file_name):
        with open(sample_directory + labels_json_file_name, 'r') as infile:
            label_file_contents = json.load(infile)
    else:
        label_file_contents = []

    labeled_files = []

    for label in label_file_contents:
        if label['filename'] in sample_files:
            labeled_files.append(label['filename'])
            sample_files.remove(label['filename'])

    # Check to see if there are unlabeled files:
    if len(sample_files) > 0:
        print('There are {} unlabeled files. Label? (y/n) '.format(len(sample_files)))
        if (input('') == 'y'):
            for sample in sample_files:
                label_dataset(sample)
                labeled_files.append(sample)
    else:
        print('No unlabeled files!')

    # Train on unlabeled files
    for label in labeled_files:
        generate_dataset(label)

    train('RNN')


