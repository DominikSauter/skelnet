import os
import shutil
import numpy as np
import scipy.ndimage
import cv2
import scipy.interpolate as interp
from skimage.transform import resize
from PIL import Image
from skimage import io
from skimage import img_as_ubyte
from skimage import filters


def main():
    in_path = 'dataset/point/'

    for path in os.listdir(in_path):
        if 'upsampled' in path and 'downsampled' not in path:# and 'test' not in path:
            print(path)
            dir_path = os.path.join(in_path, path + '_downsampled')
            shutil.rmtree(dir_path, ignore_errors=True)
            os.makedirs(dir_path)
            for img_path in os.listdir(os.path.join(in_path, path)):
                print(img_path)
                img = io.imread(os.path.join(in_path, path, img_path),as_gray=True)
                # Image.fromarray(img, 'L').show()
                img = resize(img, (256 * 1, 256 * 1), anti_aliasing=True)  # , order=1)

                thresh_img = filters.threshold_otsu(img)
                img = img >= thresh_img
                img = img.astype(float)
                io.imsave(os.path.join(dir_path, img_path), img_as_ubyte(img))


if __name__ == '__main__':
    main()
