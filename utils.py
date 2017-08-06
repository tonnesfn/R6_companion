import os
import json
import numpy as np
from PIL import Image


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


def pil_image_to_array(img):
    return np.array(img.getdata(),
                    np.uint8).reshape(img.size[1], img.size[0])


def array_to_pil_image(given_array, par1, par2=None):

    if par2 == None:
        img_to_return = Image.new("L", par1.size)
    else:
        img_to_return = Image.new("L", (par1, par2))

    img_to_return.putdata(given_array.astype(int).flatten())
    return img_to_return


# Calculate bounding box of a numpy array of an image
def bbox(given_img):
    rows = np.any(given_img, axis=1)
    cols = np.any(given_img, axis=0)
    xmin, xmax = np.where(cols)[0][[0, -1]]
    ymin, ymax = np.where(rows)[0][[0, -1]]

    return xmin, xmax, ymin, ymax
