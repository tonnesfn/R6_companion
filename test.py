from utils import get_screenshot_files
from PIL import Image
import json

import ANN
import RNN
import SVM

import processScreenshot
import ScreenshotCapture


def get_labels(given_filename):
    # Open existing json
    with open('screenshot_examples/labels.json', 'r') as infile:
        labels = json.load(infile)

    # Check if the file has already been labeled:
    for i in range(len(labels)):
        if labels[i]['filename'] == given_filename:
            return labels[i]['names']

    return None


def test_segmentation(given_filename):
    screenshot_capture = ScreenshotCapture.ScreenshotCapture()
    screenshot_example = Image.open('screenshot_examples/' + given_filename).convert('L')
    screenshot_capture.set_screenshot(screenshot_example)

    top_names, bottom_names = screenshot_capture.get_names()
    names = top_names + bottom_names
    letters = processScreenshot.process_screenshot(names)

    labels = get_labels(given_filename)

    right = 0
    wrong = 0

    for i in range(10):
        if len(labels[i]) > 0:
            if len(labels[i]) == len(letters[i]):
                right += 1
            else:
                print('    ' + labels[i] + ' not correctly classified! Got {} letters instead of correct {}'.format(len(letters[i]), len(labels[i])))
                processScreenshot.run_custom_filters(processScreenshot.threshold_image(names[i]))
#                screenshot_example.show()
#                names[i].show()
                wrong += 1

                # processScreenshot.run_custom_filters(processScreenshot.threshold_image(bottom_names[0]))

    return right, wrong

if __name__ == "__main__":
    new_files, sampled_files = get_screenshot_files()

    print('Testing on {} out of {} files.\n'.format(len(sampled_files), len(sampled_files) + len(new_files)))

    print('First testing segmentation:')
    sum_right = 0
    sum_wrong = 0

    for file in sampled_files:
        (right, wrong) = test_segmentation(file)
        sum_right += right
        sum_wrong += wrong

    print('  {} wrong of {} names'.format(sum_wrong, sum_right))
