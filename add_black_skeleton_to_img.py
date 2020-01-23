import os
from PIL import Image


def main():
    in_img_path = "dataset/point/test_ml_comp_grey"
    in_skeletonize_path = "dataset/point/skeletonize"
    out_img_path = "dataset/point/test_ml_comp_grey_skeletonize"
    
    for image, skeleton in zip(os.listdir(in_img_path), os.listdir(in_skeletonize_path)):
        img = Image.open(os.path.join(in_img_path, image), 'r')
        skel = Image.open(os.path.join(in_skeletonize_path, skeleton), 'r')
        
        # add black skeleton (skeletonize) to train image
        img_immat = img.load()
        skel_immat = skel.load()
        for i in range(256):
            for j in range(256):
                if skel_immat[(i, j)] > 0:
                    img_immat[(i, j)] = 0
        img.save(os.path.join(out_img_path, image))


if __name__ == '__main__':
    main()

