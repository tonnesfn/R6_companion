from PIL import Image
import json
import os
import lazysiege

json_file_name = 'screenshot_examples/labels.json'

directory_name = 'screenshot_examples/'
file_name = 'screenshot_2017_07_17_141006.jpg'

# Open existing json
if os.path.isfile(json_file_name):
    with open(json_file_name, 'r') as infile:
        labels = json.load(infile)
else:
    labels = []

overwrite_index = -1
# Check if the file has already been labeled:
for i in range(len(labels)):
    if labels[i]['filename'] == file_name:
        print('Filename exists, overwrite? y/n')
        if input() != 'y':
            print('Exiting program')
            exit()
        else:
            overwrite_index = i

# Open image file
current_image = Image.open(directory_name+file_name)
current_image.show()

# Check if the file has already been labeled, if so, ask to replace


# Ask for names
names = []

for i in range(10):
    while True:
        nick_input = input('>')
        returned_nick = lazysiege.lookup_player(nick_input)['player']['username']

        if returned_nick != 'notfound':
            names.append(returned_nick)
            break
        else:
            print(nick_input + ' is not a valid name! Please try again.')

# Check for valid names
if overwrite_index == -1:
    labels.append(dict(filename=file_name, names=names))
else:
    labels[overwrite_index] = dict(filename=file_name, names=names)

with open(json_file_name, 'w') as outfile:
    json.dump(labels, outfile, indent=4, sort_keys=True)
