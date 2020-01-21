import cv2
import numpy as np
import os


def main():
    in_images_path = "dataset/point/root"
    out_images_path = "dataset/point/erosion"
    
    for image_file in os.listdir(in_images_path):
        print(image_file)
        img = cv2.imread(os.path.join(in_images_path, image_file),cv2.IMREAD_GRAYSCALE)
        kernel = np.ones((2,2),np.uint8)    # set kernel size appropriately
        erosion = cv2.erode(img,kernel,iterations=1)
        cv2.imwrite(os.path.join(out_images_path, image_file), erosion)


if __name__ == '__main__':
    main()
