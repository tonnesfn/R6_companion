import os
import json
from PIL import Image

import utils
import lazysiege

sample_directory = 'screenshot_examples/'
labels_json_file_name = 'labels.json'


def label_file(given_filename):
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

if __name__ == "__main__":
    new_files, sampled_files = utils.get_screenshot_files()

    print('{} unlabeled files, {} labeled files'.format(len(new_files), len(sampled_files)))

    for filename in new_files:
        label_file(filename)