import os
import shutil
import csv
import random
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import numpy as np
import pickle
import json


def main():
    in_point_clouds_path = "dataset/point/pred"
    out_submission_path = "submission"
    skip_percent_of_points = 0

    # for every point cloud: calculate Minimum Spanning Tree
    submission = {}
    test_counter = 0
    single_adj_lists = []
    for point_cloud in os.listdir(in_point_clouds_path):
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
            
            # if image does not contain points: add some default points for default graph
            if not points:
                print(point_cloud + ' does not contain points! Some default points are used.')
                points.append(np.array([0, 0]))
                points.append(np.array([255, 255]))
                points.append(np.array([0, 255]))
                points.append(np.array([255, 0]))
                points.append(np.array([127, 127]))
                points_list.append((0, 0))
                points_list.append((255, 255))
                points_list.append((0, 255))
                points_list.append((255, 0))
                points_list.append((127, 127))

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

