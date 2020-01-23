import numpy as np
import os
from PIL import Image


def main():
    in_out_pts_npy_path = "npy/out_pts_original.npy"
    in_in_pts_npy_path = "npy/in_pts_original.npy"
    in_test_img_path = "dataset/point/test_ml_comp_grey"
    out_img_path = "dataset/point/test_ml_comp_labels_new"


    unique_entries_skelnet = []
    ml_comp = np.load('npy/in_pts.npy')
    skelnet = np.load('npy/in_pts_original.npy')
    for i in range(1219):
        print("in_pts_search: " + str(i))
        found = False
        for j in range(1000):
            pic_found = []            
            pic_found.append(np.array_equal(skelnet[i, :, :, 0], ml_comp[j, :, :, 0]))
            #for k in range(256):
            #    for l in range(256):
            #        pic_found.append(skelnet[i, k, l, 0] == ml_comp[j, k, l, 0])
            found = all(pic_found)
            if found:
                break
        if not found:
            unique_entries_skelnet.append(i)
    print(unique_entries_skelnet)


    npy_labels = np.load(in_out_pts_npy_path)
    npy_images = np.load(in_in_pts_npy_path)
    #label_indices = [2, 6, 7, 12, 19, 21, 34, 44, 49, 50, 52, 54, 63, 65, 71, 77, 81, 97, 100, 102, 110, 122, 123, 143, 144, 149, 155, 160, 162, 167, 177, 180, 186, 188, 189, 190, 204, 207, 210, 211, 215, 217,
    #        225, 227, 247, 248, 249, 250, 253, 255, 262, 264, 265, 275, 279, 280, 286, 293, 304, 307, 309, 313, 314, 338, 339, 344, 345, 354, 356, 357, 358, 359, 362, 365, 372, 375, 376, 379, 380, 385, 390, 391,
    #        396, 404, 418, 433, 434, 435, 443, 467, 468, 472, 473, 479, 490, 493, 494, 499, 506, 507, 517, 518, 524, 525, 540, 546, 550, 551, 552, 555, 559, 568, 572, 575, 578, 587, 589, 594, 595, 597, 600, 604,
    #        613, 619, 629, 630, 634, 644, 653, 655, 666, 685, 701, 719, 722, 724, 725, 728, 736, 743, 751, 752, 755, 763, 764, 767, 773, 777, 780, 781, 782, 786, 791, 792, 795, 803, 804, 808, 820, 827, 838, 850,
    #        851, 852, 858, 868, 872, 873, 878, 880, 882, 893, 904, 907, 910, 923, 927, 942, 948, 958, 959, 964, 965, 968, 972, 979, 980, 982, 990, 993, 995, 1006, 1033, 1041, 1042, 1057, 1063, 1071, 1079, 1095,
    #        1099, 1112, 1119, 1130, 1139, 1140, 1145, 1149, 1155, 1159, 1160, 1165, 1193, 1195]
    label_indices = unique_entries_skelnet

    for index in label_indices:
        npy_img = Image.fromarray(npy_images[index, :, :, 0], 'L')
        npy_img_immat = npy_img.load()
        for test_img in os.listdir(in_test_img_path):
            t_img = Image.open(os.path.join(in_test_img_path, test_img), 'r')
            t_img_immat = t_img.load()
            pic_found = []
            for i in range(256):
                for j in range(256):
                    pic_found.append(npy_img_immat[(i, j)] == t_img_immat[(i, j)])
            found = all(pic_found)
            if found:
                print("write_test_label_img: " + str(index))
                l_img = Image.fromarray(npy_labels[index, :, :, 0], 'L')
                l_img.save(os.path.join(out_img_path, test_img))
                break


if __name__ == '__main__':
    main()

