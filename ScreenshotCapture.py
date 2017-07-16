import pyscreenshot
from PIL import Image
from time import localtime, strftime, sleep
import os

name_width = 465

class ScreenshotCapture:
    raw_screenshot = None

    def get_screenshot(self):
        self.raw_screenshot = pyscreenshot.grab()
        self.raw_screenshot.save(os.path.join(os.path.curdir, 'screenshot_examples',
                                              os.getlogin() + "-%s.jpg"
                                              % strftime("%Y_%m_%d_%H%M%S", localtime())), "JPEG", quality=100)
        return self.raw_screenshot

    def get_top_scores(self):
        if self.raw_screenshot.size == (2048, 1152):
            left, top, bottom = 326, 352, 640
            cropped_image = self.raw_screenshot.crop((left, top, left+name_width, bottom))
        else:
            print('Unknown screenshot size {}{}!'.format(self.raw_screenshot.size[0],self.raw_screenshot.size[1]))

        return cropped_image

    def get_bottom_scores(self):
        if self.raw_screenshot.size == (2048, 1152):
            left, top, bottom = 326, 744, 1032
            cropped_image = self.raw_screenshot.crop((left, top, left+name_width, bottom))
            cropped_image.show()
            print('Correct size!')
        else:
            print('Unknown screenshot size {}{}!'.format(self.raw_screenshot.size[0],self.raw_screenshot.size[1]))

        return cropped_image

    def get_scores(self):
        return [self.get_top_scores(), self.get_bottom_scores()]

    def set_screenshot(self, given_image):
        self.raw_screenshot = given_image

if __name__ == "__main__":
    screenshot_capture = ScreenshotCapture()

    #sleep(8)
    #image = screenshot_capture.get_screenshot()

    screenshot_example = Image.open('screenshot_examples/T-2017_07_16_133135.jpg')
    screenshot_capture.set_screenshot(screenshot_example)

    top = screenshot_capture.get_top_scores()
    bottom = screenshot_capture.get_bottom_scores()

    print(top.size)
    print(bottom.size)
