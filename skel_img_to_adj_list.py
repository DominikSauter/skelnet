import os
import csv
import random
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import numpy as np
import pickle


def main():
    in_point_clouds_path = "dataset/point/pred"
    
    submission = {}
    test_counter = 0
    for point_cloud in os.listdir(in_point_clouds_path):
        print(point_cloud)
        with open(os.path.join(in_point_clouds_path, point_cloud)) as cloud:
            reader = csv.reader(cloud, delimiter=' ')
            points = []
            points_list = []
            for row in reader:
                # skip some % of points
                if random.randrange(0,10,1) < 5:
                   # x = float(row[0])/255
                    x = int(row[0])
                    #y = float(row[1])/255
                    y = int(row[1])
                    points.append(np.array([x, y]))
                    points_list.append((x, y))
               # print(row) 
            
            points_length = len(points)
            #graph = csr_matrix((points_length, points_length), dtype=np.float32)
            #print(graph.toarray().astype(float))
            graph = np.zeros((points_length, points_length))
            #print(graph)
            j_min = 0
            for i in range(points_length):
                for j in range(j_min, points_length):
                    graph[i][j] = np.sum(np.absolute(points[i] - points[j]))
                j_min += 1
            #print(graph)

            graph = csr_matrix(graph)
            mst = minimum_spanning_tree(graph)
            #print(mst.toarray().astype(float))
            
            mst = mst.toarray().astype(float)
            #print(mst)
            mst_full_matrix = mst + mst.transpose()
            #print(mst_full_matrix)
            adjacency = []
            for i in range(points_length):
                adjacency.append([])
                for j in range(points_length):
                    if mst_full_matrix[i][j] > 0:
                         adjacency[i].append(j)

            points_list = [(x/255.0, y/255.0) for (x, y) in points_list]

            submission[point_cloud[:-3]] = {'adjacency': adjacency,
                                            'coordinates': points_list
                                            }
            #if test_counter == 1:
             #   submission = {'adjacency': adjacency,
              #                  'coordinates': points_list
               #                 }
                #test_submission = {'adjacency': [[1], [0, 1], [0, 3], [2]],
                 #                   'coordinates': [(0.2, 0.3), (0.4, 0.5), (0.9, 0.8), (0.1, 0.1)]
                  #                  }
                #break
            #test_counter += 1
        
            #print(submission.values())
    with open("submission/submission.pkl", "wb") as f:
        pickle.dump(submission, f)

    with open("submission/submission.pkl", "rb") as f:
        loaded = pickle.load(f)
        print(loaded)
    #with open("submission/sample_submission_12.243.pkl", "rb") as f:
        #print(pickle.load(f).values())
        


if __name__ == '__main__':
        main()
