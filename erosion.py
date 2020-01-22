import cv2
import numpy as np
import os


def main():
    in_images_path = "dataset/point/merged"
    out_images_path = "dataset/point/erosion"
    
    for image_file in os.listdir(in_images_path):
        print(image_file)
        img = cv2.imread(os.path.join(in_images_path, image_file),cv2.IMREAD_GRAYSCALE)
        kernel_dilation = np.ones((2,2),np.uint8)    # set kernel size appropriately
        kernel_erosion = np.ones((2,2),np.uint8)    # set kernel size appropriately
        
        img = cv2.dilate(img,kernel_dilation,iterations=1)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_dilation)
        img = cv2.erode(img,kernel_erosion,iterations=2)
        
        cv2.imwrite(os.path.join(out_images_path, image_file), img)


if __name__ == '__main__':
    main()
