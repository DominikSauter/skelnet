import os
import shutil
import csv
import random
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import numpy as np
import pickle
import json
from PIL import Image


def main():
    in_point_clouds_path = "dataset/point/pred"
    out_submission_path = "submission"
    skip_percent_of_points = 0

    default_pt = [(4, 4), (4, 251), (5, 5), (5, 250), (6, 6), (6, 249), (7, 7), (7, 248), (8, 8), (8, 247), (9, 9), (9, 246), (10, 10), (10, 245), (11, 11), (11, 244), (12, 12), (12, 243), (13, 13),
            (13, 242), (14, 14), (14, 241), (15, 15), (15, 240), (16, 16), (16, 239), (17, 17), (17, 238), (18, 18), (18, 237), (19, 19), (19, 236), (20, 20), (20, 235), (21, 21), (21, 234), (22, 22),
            (22, 233), (23, 23), (23, 232), (24, 24), (24, 231), (25, 25), (25, 230), (26, 26), (26, 229), (27, 27), (27, 228), (28, 28), (28, 227), (29, 29), (29, 226), (30, 30), (30, 225), (31, 31),
            (31, 224), (32, 32), (32, 223), (33, 33), (33, 222), (34, 34), (34, 221), (35, 35), (35, 220), (36, 36), (36, 219), (37, 37), (37, 218), (38, 38), (38, 217), (39, 39), (39, 216), (40, 40),
            (40, 215), (41, 41), (41, 214), (42, 42), (42, 213), (43, 43), (43, 212), (44, 44), (44, 211), (45, 45), (45, 210), (46, 46), (46, 209), (47, 47), (47, 208), (48, 48), (48, 207), (49, 49),
            (49, 206), (50, 50), (50, 205), (51, 51), (51, 204), (52, 52), (52, 203), (53, 53), (53, 202), (54, 54), (54, 201), (55, 55), (55, 200), (56, 56), (56, 199), (57, 57), (57, 198), (58, 58),
            (58, 197), (59, 59), (59, 196), (60, 60), (60, 195), (61, 61), (61, 194), (62, 62), (62, 193), (63, 63), (63, 192), (64, 64), (64, 191), (65, 65), (65, 190), (66, 66), (66, 189), (67, 67),
            (67, 188), (68, 68), (68, 187), (69, 69), (69, 186), (70, 70), (70, 185), (71, 71), (71, 184), (72, 72), (72, 183), (73, 73), (73, 182), (74, 74), (74, 181), (75, 75), (75, 180), (76, 76),
            (76, 179), (77, 77), (77, 178), (78, 78), (78, 177), (79, 79), (79, 176), (80, 80), (80, 175), (81, 81), (81, 174), (82, 82), (82, 173), (83, 83), (83, 172), (84, 84), (84, 171), (85, 85),
            (85, 170), (86, 86), (86, 169), (87, 87), (87, 168), (88, 88), (88, 167), (89, 89), (89, 166), (90, 90), (90, 165), (91, 91), (91, 164), (92, 92), (92, 163), (93, 93), (93, 162), (94, 94),
            (94, 161), (95, 95), (95, 160), (96, 96), (96, 159), (97, 97), (97, 158), (98, 98), (98, 157), (99, 99), (99, 156), (100, 100), (100, 155), (101, 101), (101, 154), (102, 102), (102, 153),
            (103, 103), (103, 152), (104, 104), (104, 105), (104, 106), (104, 107), (104, 108), (104, 109), (104, 110), (104, 111), (104, 112), (104, 113), (104, 114), (104, 115), (104, 116), (104, 117),
            (104, 118), (104, 119), (104, 120), (104, 121), (104, 122), (104, 123), (104, 124), (104, 125), (104, 126), (104, 127), (104, 128), (104, 129), (104, 130), (104, 131), (104, 132), (104, 133),
            (104, 134), (104, 135), (104, 136), (104, 137), (104, 138), (104, 139), (104, 140), (104, 141), (104, 142), (104, 143), (104, 144), (104, 145), (104, 146), (104, 147), (104, 148), (104, 149),
            (104, 150), (104, 151), (105, 103), (105, 152), (106, 102), (106, 153), (107, 101), (107, 154), (108, 100), (108, 155), (109, 99), (109, 156), (110, 98), (110, 157), (111, 97), (111, 158),
            (112, 96), (112, 159), (113, 95), (113, 160), (114, 94), (114, 161), (115, 93), (115, 162), (116, 92), (116, 163), (117, 91), (117, 164), (118, 90), (118, 165), (119, 89), (119, 166),
            (120, 88), (120, 167), (121, 87), (121, 168), (122, 86), (122, 169), (123, 85), (123, 170), (124, 84), (124, 171), (125, 83), (125, 172), (126, 82), (126, 173), (127, 81), (127, 174),
            (128, 80), (128, 175), (129, 79), (129, 176), (130, 78), (130, 177), (131, 77), (131, 178), (132, 76), (132, 179), (133, 75), (133, 180), (134, 74), (134, 181), (135, 73), (135, 182),
            (136, 72), (136, 183), (137, 71), (137, 184), (138, 70), (138, 185), (139, 69), (139, 186), (140, 68), (140, 187), (141, 67), (141, 188), (142, 66), (142, 189), (143, 65), (143, 190),
            (144, 64), (144, 191), (145, 63), (145, 192), (146, 62), (146, 193), (147, 61), (147, 194), (148, 60), (148, 195), (149, 59), (149, 196), (150, 58), (150, 197), (151, 57), (151, 198),
            (152, 56), (152, 199), (153, 55), (153, 200), (154, 54), (154, 201), (155, 53), (155, 202), (156, 52), (156, 203), (157, 51), (157, 204), (158, 50), (158, 205), (159, 49), (159, 206),
            (160, 48), (160, 207), (161, 47), (161, 208), (162, 46), (162, 209), (163, 45), (163, 210), (164, 44), (164, 211), (165, 43), (165, 212), (166, 42), (166, 213), (167, 41), (167, 214),
            (168, 40), (168, 215), (169, 39), (169, 216), (170, 38), (170, 217), (171, 37), (171, 218), (172, 36), (172, 219), (173, 35), (173, 220), (174, 34), (174, 221), (175, 33), (175, 222),
            (176, 32), (176, 223), (177, 31), (177, 224), (178, 30), (178, 225), (179, 29), (179, 226), (180, 28), (180, 227), (181, 27), (181, 228), (182, 26), (182, 229), (183, 25), (183, 230),
            (184, 24), (184, 231), (185, 23), (185, 232), (186, 22), (186, 233), (187, 21), (187, 234), (188, 20), (188, 235), (189, 19), (189, 236), (190, 18), (190, 237), (191, 17), (191, 238),
            (192, 16), (192, 239), (193, 15), (193, 240), (194, 14), (194, 241), (195, 13), (195, 242), (196, 12), (196, 243), (197, 11), (197, 244), (198, 10), (198, 245), (199, 9), (199, 246),
            (200, 8), (200, 247), (201, 7), (201, 248), (202, 6), (202, 249), (203, 5), (203, 250), (204, 4), (204, 251)]

    # for every point cloud: calculate Minimum Spanning Tree
    submission = {}
    test_counter = 0
    single_adj_lists = []
    for point_cloud in sorted(os.listdir(in_point_clouds_path)):
        print(point_cloud)
        with open(os.path.join(in_point_clouds_path, point_cloud)) as cloud:
            reader = csv.reader(cloud, delimiter=' ')
            points = []
            points_list = []
            for row in reader:
                # skip some % of points
                if random.randrange(0,100,1) >= skip_percent_of_points:
                    x = int(row[0])
                    y = int(row[1])
                    points.append(np.array([x, y]))
                    points_list.append((x, y))

            print(len(points)/(256*256))
            # if image does not contain points: add some default points for default graph
            if not points or len(points)/(256*256) > 0.1:
                print(len(points)/(256*256))
                points = []
                points_list = []
                #for pt in default_pt:
                #    points.append(np.array([pt[0], pt[1]]))
                #    points_list.append(pt)

                for i in range(64, 192, 2):
                    points.append(np.array([127, i]))
                    points_list.append((127, i))
                    points.append(np.array([i, 127]))
                    points_list.append((i, 127))

                    points.append(np.array([i, i]))
                    points_list.append((i, i))
                    points.append(np.array([i, -i+256]))
                    points_list.append((i, -i+256))

                #points_img = Image.new('L', (256, 256))
                #points_immat = points_img.load()
                #for point in points_list:
                #    points_immat[int(point[0]), int(point[1])] = 255
                #Image.fromarray(np.array(points_img), 'L').show()
                #break

            # preparation Minimum Spanning Tree
            points_length = len(points)
            graph = np.zeros((points_length, points_length))
            j_min = 0
            for i in range(points_length):
                for j in range(j_min, points_length):
                    graph[i][j] = np.sum(np.absolute(points[i] - points[j]))
                j_min += 1
            # calculate Minimum Spanning Tree
            graph = csr_matrix(graph)
            mst = minimum_spanning_tree(graph)
            # create adjacency information
            mst = mst.toarray().astype(float)
            mst_full_matrix = mst + mst.transpose()
            adjacency = []
            for i in range(points_length):
                adjacency.append([])
                for j in range(points_length):
                    if mst_full_matrix[i][j] > 0:
                         adjacency[i].append(j)
            # normalize points and change x and y and mirror x (with offset to get positive value)
            points_list = [(y/255.0, (-x/255.0)+1) for (x, y) in points_list]
            # store results
            single_adj_list = {'adjacency': adjacency,
                                'coordinates': points_list
                                }
            submission[point_cloud[:-3]] = single_adj_list
            single_adj_lists.append((single_adj_list, point_cloud))

    # delete out dirs and recreate    
    shutil.rmtree(out_submission_path, ignore_errors=True)
    single_adj_lists_path = os.path.join(out_submission_path, 'single_adj_lists')
    if not os.path.isdir(single_adj_lists_path):
        os.makedirs(single_adj_lists_path)

    # write to file
    with open("submission/submission.pkl", "wb") as f:
        pickle.dump(submission, f)
    with open("submission/submission.json", "w+") as f:
        json.dump(submission, f, indent=4)
    for adj_list in single_adj_lists:
        with open(os.path.join("submission/single_adj_lists", adj_list[1] + ".pkl"), "wb") as f:
            pickle.dump(adj_list[0], f)
    # read again and print
    with open("submission/submission.pkl", "rb") as f:
        loaded = pickle.load(f)
        print(loaded)        


if __name__ == '__main__':
    main()

