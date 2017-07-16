import processScreenshot
from PIL import Image
import json
import os.path
import ScreenshotCapture

current_file = 'screenshot_examples/bottomtest_2017_07_15_162346.jpg'

alreadyTrainedOn = []

if os.path.isfile('dataset/readImages.json') == True:
    with open('dataset/readImages.json') as f:
        alreadyTrainedOn = json.load(f)


if current_file in alreadyTrainedOn:
    print('Already generated samples from this file!')
else:

    character_dataset = processScreenshot.CharacterDataset()
    character_dataset.load_data_set('dataset/character_samples.pickle')
    character_dataset.currently_training = True

    screenshot_capture = ScreenshotCapture.ScreenshotCapture()
    screenshot_example = Image.open('screenshot_examples/T-2017_07_16_133135.jpg')
    screenshot_capture.set_screenshot(screenshot_example)
    top_image, bottom_image = screenshot_capture.get_names()

    sentence_images = processScreenshot.process_screenshot(top_image.convert('L'))
    nicks = processScreenshot.CharacterDataset.classify_sentences(character_dataset, sentence_images)

    character_dataset.save_data_set('dataset')

    alreadyTrainedOn.append(current_file)
    with open('dataset/readImages.json', 'w') as outfile:
        json.dump(alreadyTrainedOn, outfile, indent=4, sort_keys=True)