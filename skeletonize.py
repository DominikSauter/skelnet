from skimage.morphology import skeletonize
from skimage import data
import matplotlib.pyplot as plt
from skimage.util import invert
import os
from PIL import Image
import numpy as np


def main():
    in_images = "dataset/point/test_ml_comp_grey"
    out_images = "dataset/point/root"

    for image in os.listdir(in_images):
        print(image)
        img = Image.open(os.path.join(in_images, image), 'r')
        img_npy = np.array(img)
        img_npy = (img_npy > .5).astype(float)
        # perform skeletonization
        skeleton = skeletonize(img_npy)
        skel_img = Image.fromarray(skeleton)
        skel_img.save(os.path.join(out_images, image))

        # display results
        #fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
        #                        sharex=True, sharey=True)
        #ax = axes.ravel()
        #
        #ax[0].imshow(img_npy, cmap=plt.cm.gray)
        #ax[0].axis('off')
        #ax[0].set_title('original', fontsize=20)
        #
        #ax[1].imshow(skeleton, cmap=plt.cm.gray)
        #ax[1].axis('off')
        #ax[1].set_title('skeleton', fontsize=20)
        #
        #fig.tight_layout()
        #plt.show()


if __name__ == '__main__':
    main()

