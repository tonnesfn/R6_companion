import ScreenshotCapture
from PIL import Image

import processScreenshot
import ANN


class Classifier:
    ann = ANN.ANN()

    def get_names(self, screenshot):
        # Screenshot version:
        screenshot_capture = ScreenshotCapture.ScreenshotCapture()
        screenshot_capture.set_screenshot(screenshot)

        top_names, bottom_names = screenshot_capture.get_names()

        letter_images = processScreenshot.get_letters(top_names, bottom_names)

        names = []
        for sentence in letter_images:
            names.append(''.join(self.ann.get_prediction(sentence)))

        return names

if __name__ == "__main__":
    classifier = Classifier()
    screenshot_example = Image.open('screenshot_examples/screenshot_2017_07_17_185104.jpg').convert('L')

    screenshot_example.show()
    names = classifier.get_names(screenshot_example)
    print(names)
