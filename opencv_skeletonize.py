import os
import shutil
import cv2
import numpy as np


def main():
    in_img_path = 'dataset/point/test_ml_comp_grey'
    out_img_path = 'dataset/point/opencv_thinned'
    out_img_path_theailearner = 'dataset/point/opencv_thinned_theailearner'
    out_img_path_jsheedy = 'dataset/point/opencv_thinned_jsheedy'

    # delete out dirs and recreate
    shutil.rmtree(out_img_path, ignore_errors=True)
    if not os.path.isdir(out_img_path):
        os.makedirs(out_img_path)
    # delete out dirs and recreate    
    shutil.rmtree(out_img_path_theailearner, ignore_errors=True)
    if not os.path.isdir(out_img_path_theailearner):
        os.makedirs(out_img_path_theailearner)
    # delete out dirs and recreate    
    shutil.rmtree(out_img_path_jsheedy, ignore_errors=True)
    if not os.path.isdir(out_img_path_jsheedy):
        os.makedirs(out_img_path_jsheedy)

    for img_path in sorted(os.listdir(in_img_path)):
        print(img_path)
        img_original = cv2.imread(os.path.join(in_img_path,img_path),0)


        # STACKOVERFLOW
        img = img_original.copy()
        size = np.size(img)
        skel = np.zeros(img.shape,np.uint8)

        ret,img = cv2.threshold(img,127,255,0)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        done = False

        while(not done):
            eroded = cv2.erode(img,element)
            temp = cv2.dilate(eroded,element)
            temp = cv2.subtract(img,temp)
            skel = cv2.bitwise_or(skel,temp)
            img = eroded.copy()

            zeros = size - cv2.countNonZero(img)
            if zeros==size:
                done = True

        cv2.imwrite(os.path.join(out_img_path,img_path),skel) 
        
        #cv2.imshow("skel",skel)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        #break



        # THEAILEARNER
        img1 = img_original.copy()

        # Structuring Element
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        # Create an empty output image to hold values
        thin = np.zeros(img.shape,dtype='uint8')

        # Loop until erosion leads to an empty set
        while (cv2.countNonZero(img1)!=0):
            # Erosion
            erode = cv2.erode(img1,kernel)
            # Opening on eroded image
            opening = cv2.morphologyEx(erode,cv2.MORPH_OPEN,kernel)
            # Subtract these two
            subset = erode - opening
            # Union of all previous sets
            thin = cv2.bitwise_or(subset,thin)
            # Set the eroded image for next iteration
            img1 = erode.copy()

        #cv2.imshow('original',img)
        #cv2.imshow('thinned',thin)
        #cv2.waitKey(0)

        cv2.imwrite(os.path.join(out_img_path_theailearner,img_path),thin)



        # JSHEEDY
        """ OpenCV function to return a skeletonized version of img, a Mat object"""
        #  hat tip to http://felix.abecassis.me/2011/09/opencv-morphological-skeleton/

        img = img_original.copy() # don't clobber original
        skel = img.copy()

        skel[:,:] = 0
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

        while True:
            eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
            temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
            temp  = cv2.subtract(img, temp)
            skel = cv2.bitwise_or(skel, temp)
            img[:,:] = eroded[:,:]
            if cv2.countNonZero(img) == 0:
                break

        cv2.imwrite(os.path.join(out_img_path_jsheedy,img_path),skel)



if __name__ == '__main__':
    main()

