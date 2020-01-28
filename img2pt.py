'''Utility for converting from image to point cloud

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
import sys

import os
import shutil
import datetime
import scipy.misc
import json
from skimage.io import imsave

import matplotlib.pyplot as plt

from utils import read_points
from utils import list_files, read_gray

ROOT_PATH = "dataset/point/erosion"
PRED_PATH = "dataset/point/pred"

def img2pt():
    shutil.rmtree(PRED_PATH, ignore_errors=True)

    path = ROOT_PATH
    if not os.path.isdir(PRED_PATH):
        os.makedirs(PRED_PATH)
    files = list_files(path)
    pix = []
    for f in files:
        pix_file = os.path.join(path, f)
        print(pix_file)
        filename = pix_file
        filename = filename.replace(".png", "")
        filename = filename.replace("full", "skel")
        #filename = filename.replace(os.path.basename(os.path.normpath(ROOT_PATH)), "pred")
        filename = os.path.join(PRED_PATH, os.path.basename(os.path.normpath(filename)))
        print(filename)
        pix_data =  read_gray(pix_file)
        with open(filename, "w+") as  fh:
            for x in range(pix_data.shape[0]):
                for y in range(pix_data.shape[0]):
                    if pix_data[x][y]>0:
                        fh.write("%d %d\n" % (x, y))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path",
                        default=ROOT_PATH)
    args = parser.parse_args()
    ROOT_PATH = args.in_path
    img2pt()
