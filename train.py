import processScreenshot
from PIL import Image
import json
import os.path
import ScreenshotCapture

current_file = 'screenshot_examples/screenshot_2017_07_17_133442.jpg'

alreadyTrainedOn = []

if os.path.isfile('dataset/readImages.json') == True:
    with open('dataset/readImages.json') as f:
        alreadyTrainedOn = json.load(f)


if current_file in alreadyTrainedOn:
    print('Already generated samples from this file!')
else:

    screenshot_capture = ScreenshotCapture.ScreenshotCapture()
    screenshot_example = Image.open(current_file).convert('L')
    screenshot_capture.set_screenshot(screenshot_example)

    top_names, bottom_names = screenshot_capture.get_names()

    processScreenshot.get_nicks(top_names, True)
    processScreenshot.get_nicks(bottom_names, True)

    alreadyTrainedOn.append(current_file)
    with open('dataset/readImages.json', 'w') as outfile:
        json.dump(alreadyTrainedOn, outfile, indent=4, sort_keys=True)