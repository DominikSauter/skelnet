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
    in_img_path = 'dataset/point/test_ml_comp_grey'
    out_img_path = 'dataset/point/test_ml_comp_grey_upsampled'

    # delete out dirs and recreate
    shutil.rmtree(out_img_path, ignore_errors=True)
    os.makedirs(out_img_path)

    for img_path in sorted(os.listdir(in_img_path)):
        print(img_path)
        # img = cv2.imread(os.path.join(in_img_path, img_path),cv2.IMREAD_GRAYSCALE)
        #
        # # img_row = img[:, 0]
        # # img_col = img[0, :]
        # # f = interp.RectBivariateSpline(img_row, img_col, img, kx=1, ky=1)
        # # new_im = f(new_x, new_y)
        #
        # #'Resampled by a factor of 2 with cubic interpolation:'
        # img = scipy.ndimage.zoom(img, 2, order=3)
        # cv2.imwrite(os.path.join(out_img_path, img_path), img)

        img = io.imread(os.path.join(in_img_path, img_path))
        img = resize(img, (256 * 0.5, 256 * 0.5), anti_aliasing=True)#, order=1)

        thresh_img = filters.threshold_otsu(img)
        img = img >= thresh_img
        img = img.astype(float)
        io.imsave(os.path.join(out_img_path, img_path), img_as_ubyte(img))


    # x = np.arange(9).reshape(3, 3)
    # #'Resampled by a factor of 2 with nearest interpolation:'
    # scipy.ndimage.zoom(x, 2, order=0)
    # #'Resampled by a factor of 2 with bilinear interpolation:'
    # scipy.ndimage.zoom(x, 2, order=1)
    # #'Resampled by a factor of 2 with cubic interpolation:'
    # scipy.ndimage.zoom(x, 2, order=3)


if __name__ == '__main__':
    main()
