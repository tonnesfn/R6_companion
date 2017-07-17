import processScreenshot
from PIL import Image
import json
import os.path
import ScreenshotCapture

json_file_name = 'screenshot_examples/labels.json'
sample_directory = 'screenshot_examples/'
current_file = 'screenshot_2017_07_17_141006.jpg'

alreadyTrainedOn = []

if os.path.isfile('dataset/readImages.json') == True:
    with open('dataset/readImages.json') as f:
        alreadyTrainedOn = json.load(f)

if (sample_directory+current_file) in alreadyTrainedOn:
    print('Already generated samples from this file!')
else:
    # Open existing json
    if os.path.isfile(json_file_name):
        with open(json_file_name, 'r') as infile:
            labels = json.load(infile)
    else:
        labels = []

    image_labels = []
    # Check if the file has already been labeled:
    for i in range(len(labels)):
        if labels[i]['filename'] == current_file:
            labeled = True
            image_labels = labels[i]['names']

    if len(image_labels) == 0:
        print('Current image has not been labeled!')
        exit()

    screenshot_capture = ScreenshotCapture.ScreenshotCapture()
    screenshot_example = Image.open(sample_directory + current_file).convert('L')
    screenshot_capture.set_screenshot(screenshot_example)

    top_names, bottom_names = screenshot_capture.get_names()

    processScreenshot.get_nicks(top_names, image_labels[:5])
    processScreenshot.get_nicks(bottom_names, image_labels[5:])

    alreadyTrainedOn.append(sample_directory + current_file)
    with open('dataset/readImages.json', 'w') as outfile:
        json.dump(alreadyTrainedOn, outfile, indent=4, sort_keys=True)
