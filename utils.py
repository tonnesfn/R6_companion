import os
import json


def save_pil_image(given_image, filename):
    given_image.save(filename, "JPEG", quality=100)

def get_screenshot_files():
    new_files = os.listdir('screenshot_examples/')
    new_files.remove('labels.json')

    label_file_contents = []
    if os.path.isfile('screenshot_examples/labels.json'):
        with open('screenshot_examples/labels.json', 'r') as infile:
            label_file_contents = json.load(infile)
    else:
        label_file_contents = []

    sampled_files = []

    for label in label_file_contents:
        if label['filename'] in new_files:
            sampled_files.append(label['filename'])
            new_files.remove(label['filename'])

    return new_files, sampled_files
