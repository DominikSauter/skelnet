import numpy as np
from PIL import Image
import os

IN_IMAGES_PATH = "/home/DominikSauter/skelnet/dataset/point/test_rgb/"
OUT_IMAGES_PATH = "/home/DominikSauter/skelnet/dataset/point/test_grey/"

for image in os.listdir(IN_IMAGES_PATH):
    print(image)
    in_path = os.path.join(IN_IMAGES_PATH, image)
    out_path = os.path.join(OUT_IMAGES_PATH, image)
    x = Image.open(in_path, 'r')
    x = x.convert('L') # makes it greyscale
    x.save(out_path)
