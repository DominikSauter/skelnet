import numpy as np
from PIL import Image
import os
import random

IN_IMAGES_PATH = "/home/DominikSauter/skelnet/dataset/point/test_ml_comp_grey/"
OUT_IMAGES_PATH = "/home/DominikSauter/skelnet/dataset/point/test_ml_comp_noise/"

for image in os.listdir(IN_IMAGES_PATH):
    print(image)
    in_path = os.path.join(IN_IMAGES_PATH, image)
    out_path = os.path.join(OUT_IMAGES_PATH, image)
    x = Image.open(in_path, 'r')
    pixels = x.load() # create the pixel map
    for i in range(x.size[0]): # for every pixel:
        for j in range(x.size[1]):
            if random.random() < 0.5:
                pixels[i, j] = 0
    x.save(out_path)

