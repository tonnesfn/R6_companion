import pyscreenshot
from PIL import Image
from time import localtime, strftime, sleep
import os

name_width = 465
left_offset = 42


class ScreenshotCapture:
    raw_screenshot = None

    def get_screenshot(self):
        self.raw_screenshot = pyscreenshot.grab()
        self.raw_screenshot.save(os.path.join(os.path.curdir, 'screenshot_examples',
                                              os.getlogin() + "-%s.jpg"
                                              % strftime("%Y_%m_%d_%H%M%S", localtime())), "JPEG", quality=100)
        return self.raw_screenshot

    def get_top_names(self):
        if self.raw_screenshot.size == (2048, 1152):
            # These numbers should result in the full black box only (excluding white line to the left)
            left, tops = 322, [351, 410, 469, 527, 586]
            line_height = 53
        elif self.raw_screenshot.size == (2560, 1440):
            left, tops = 412, [439, 512, 585, 659, 732]
            line_height = 67
        else:
            print('Unknown screenshot size {}{}!'.format(self.raw_screenshot.size[0],self.raw_screenshot.size[1]))

        lines = []
        for i in range(5):
            lines.append(self.raw_screenshot.crop((left+left_offset, tops[i], left+name_width, tops[i]+line_height)))
            #lines[-1].show()

        return lines

    def get_bottom_names(self):
        if self.raw_screenshot.size == (2048, 1152):
            # These numbers should result in the full black box only (excluding white line to the left)
            left, tops = 322, [744, 802, 861, 920, 978]
            line_height = 53
        elif self.raw_screenshot.size == (2560, 1440):
            left, tops = 408, [929, 1003, 1076, 1149, 1223]
            line_height = 67
        else:
            print('Unknown screenshot size {}{}!'.format(self.raw_screenshot.size[0],self.raw_screenshot.size[1]))

        lines = []
        for i in range(5):
            lines.append(self.raw_screenshot.crop((left+left_offset, tops[i], left+name_width, tops[i]+line_height)))
            #lines[-1].show()

        return lines

    def get_names(self):
        return [self.get_top_names(), self.get_bottom_names()]

    def set_screenshot(self, given_image):
        self.raw_screenshot = given_image

if __name__ == "__main__":
    screenshot_capture = ScreenshotCapture()

    sleep(4)
    image = screenshot_capture.get_screenshot()

    #screenshot_example = Image.open('screenshot_examples/T-2017_07_16_133135.jpg')
    #screenshot_capture.set_screenshot(screenshot_example)

    #top = screenshot_capture.get_top_names()
    #bottom = screenshot_capture.get_bottom_names()

    #print(top.size)
    #print(bottom.size)
