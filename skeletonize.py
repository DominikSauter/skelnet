from skimage.morphology import skeletonize, medial_axis, thin
from skimage import data
import matplotlib.pyplot as plt
from skimage.util import invert
import os
import shutil
from PIL import Image
import numpy as np


def main():
    in_images = "dataset/point/test_ml_comp_grey_upsampled_downsampled"
    out_images_skel = "dataset/point/skeletonize_testupdownsampled"
    out_images_skel_lee = "dataset/point/skeletonize_lee_testupdownsampled"
    out_images_skel_medial_axis = "dataset/point/medial_axis_testupdownsampled"
    out_images_skel_thinned = "dataset/point/thinned_testupdownsampled"

    shutil.rmtree(out_images_skel, ignore_errors=True)
    os.makedirs(out_images_skel, exist_ok=True)
    shutil.rmtree(out_images_skel_lee, ignore_errors=True)
    os.makedirs(out_images_skel_lee, exist_ok=True)
    shutil.rmtree(out_images_skel_medial_axis, ignore_errors=True)
    os.makedirs(out_images_skel_medial_axis, exist_ok=True)
    shutil.rmtree(out_images_skel_thinned, ignore_errors=True)
    os.makedirs(out_images_skel_thinned, exist_ok=True)

    for image in os.listdir(in_images):
        print(image)
        img = Image.open(os.path.join(in_images, image), 'r')
        img_npy = np.array(img)
        img_npy = (img_npy > .5).astype(float)
        # perform skeletonization
        skeleton = skeletonize(img_npy)
        skeleton_lee = skeletonize(img_npy, method = 'lee')
        # Compute the medial axis (skeleton) and the distance transform
        skeleton_medial_axis, distance = medial_axis(img_npy, return_distance=True)
        skeleton_thinned = thin(img_npy)

        skel_img = Image.fromarray(skeleton)
        skel_img_lee = Image.fromarray(skeleton_lee)
        skel_img_medial_axis = Image.fromarray(skeleton_medial_axis)
        skel_img_thinned = Image.fromarray(skeleton_thinned)
        
        skel_img.save(os.path.join(out_images_skel, image))
        skel_img_lee.save(os.path.join(out_images_skel_lee, image))
        skel_img_medial_axis.save(os.path.join(out_images_skel_medial_axis, image))
        skel_img_thinned.save(os.path.join(out_images_skel_thinned, image))

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

