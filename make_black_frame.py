import os
import shutil
import numpy as np
from PIL import Image


def main():
    in_img_path = "dataset/point/test_ml_comp_grey"
    out_img_path = "dataset/point/test_ml_comp_grey_framed"

    shutil.rmtree(out_img_path, ignore_errors=True)
    os.makedirs(out_img_path, exist_ok=True)

    counter = 0
    for img_path in sorted(os.listdir(in_img_path)):
        img = Image.open(os.path.join(in_img_path, img_path), 'r')
        img_immat = img.load()
        
        for i in range(256):
            img_immat[(i, 0)] = 0
            img_immat[(0, i)] = 0
            #img_immat[(i, 1)] = 255
            #img_immat[(1, i)] = 255
            #img_immat[(i, 2)] = 255
            #img_immat[(2, i)] = 255
            
            img_immat[(i, 255)] = 0
            img_immat[(255, i)] = 0
            #img_immat[(i, 254)] = 255
            #img_immat[(254, i)] = 255
            #img_immat[(i, 253)] = 255
            #img_immat[(253, i)] = 255
        img.save(os.path.join(out_img_path, img_path))
        
        #if counter == 43:    
        #    Image.fromarray(np.array(img), 'L').show()
        #    for i in range(256):
        #        for j in range(256):
        #            img_immat[(i, j)] = 0
        #    break
        counter += 1


if __name__ == '__main__':
    main()

