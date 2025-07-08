import json
import numpy as np
import os
from PIL import Image

TRAIN_DIR = "dataset"
if not os.path.isdir(TRAIN_DIR):
    os.mkdir(TRAIN_DIR)

img = np.random.randint(0, 50, [100000, 64, 64], dtype=np.uint8)
square = np.random.randint(100, 200, [100000, 15, 15], dtype=np.uint8)

coords = np.empty([100000, 2])
data = {}

for i in range(img.shape[0]):

    x = np.random.randint(20, 44)
    y = np.random.randint(20, 44)

    img[i, (y - 7): (y + 8), (x - 7): (x + 8)] = square[i]
    coords[i] = [y, x]

    name_img = f'img_{i}.jpeg'
    path_img = os.path.join(TRAIN_DIR, "train", name_img)

    image = Image.fromarray(img[i])
    image.save(path_img)

    data[name_img] = [y, x]
    with open(os.path.join(TRAIN_DIR, "coords.json"), 'w') as f:
        json.dump(data, f, indent=2)