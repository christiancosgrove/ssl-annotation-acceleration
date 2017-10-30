from data import DataReader
import matplotlib.pyplot as plt

from model import SSLModel

import time
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_dir', dest='CHECKPOINT_DIR', nargs='?', const='checkpoints_4000_unsupervised')
args = parser.parse_args()

mb_size = 32
images_directory = 'images'
width = 32
channels = 3

chunk_size = 100

from numpy import genfromtxt
class_list = [x.decode('ascii') for x in genfromtxt('classes.csv', delimiter=',', dtype=None)]

import os

CLUSTER_DEPTH = 8

def main():
    reader = DataReader(images_directory, width, width, channels, class_list)

    model = SSLModel(width, width, channels, mb_size, len(class_list), args.CHECKPOINT_DIR, load=True)

    #perform clustering
    reader.autolabel(model, 1.1, use_clustering=True)

    levels = len(reader.image_list[0].clusters)


    cor, tot = reader.evaluate_model(model)

    print("test performance {}".format(cor/tot))

    #all possible cluster ranges
    #first two ranges must have a 
    sizes = []
    for i in range(CLUSTER_DEPTH):
        sizes.append(2**(i+1))


    evals = []

    for clustering_level in range(CLUSTER_DEPTH):

        indices = np.random.permutation(len(reader.image_list))

        # randomly sample groups of images of size 32
        max_group_size = 64

        print("clustering level {}".format(clustering_level))

        n_interactions = 0
        n_correct = 0
        n_total = 0
        def evaluate_group(group, max_clustering_level, curr_clustering_level=0):
            if len(group) == 0:
                return 0, 0, 0
            incorrect = 0
            for ind,cluster in group:
                if reader.image_list[ind].ground_truth != cluster[0]:
                    incorrect += 1
                    break


            if incorrect == 0:
                return 1, len(group), len(group)
            else:
                if curr_clustering_level < max_clustering_level:

                    new_evals = [evaluate_group([x for x in group if x[1][curr_clustering_level+2] == c], curr_clustering_level + 1, max_clustering_level) for c in range(sizes[curr_clustering_level])]

                    return tuple(np.sum(new_evals,axis=0))

                else:

                    n_interactions = 0
                    n_correct = 0
                    n_total = 0

                    #group[0][1][0] is a surrogate for the proposed label
                    for ind, clusters in group:
                        if reader.image_list[ind].ground_truth == group[0][1][0]:
                            n_correct+=1
                        n_total+=1
                    n_interactions +=1

                    #if a majority of the images are not correct, automatically label them as negatives
                    if n_correct < len(group) / 2:
                        n_correct = len(group) - n_correct

                    return n_interactions, n_correct, n_total

        for c in range(len(class_list)):
            for k_cluster in range(sizes[1]):
                for a_cluster in range(sizes[clustering_level]):
                    i = 0
                    group = []
                    while i < len(indices):
                        if reader.image_list[indices[i]].clusters[0] == c:
                            if (reader.image_list[indices[i]].clusters[1] == k_cluster):
                                if (reader.image_list[indices[i]].clusters[clustering_level+2] == a_cluster):
                                    group.append((indices[i],reader.image_list[indices[i]].clusters[:2+clustering_level]))

                        i+=1

                        if len(group) >= max_group_size:
                            inter, corr, tot = evaluate_group(group, clustering_level)
                            n_interactions+=inter
                            n_correct += corr
                            n_total += tot
                            # print('appendign group c: {} k {} a {}'.format(c, k_cluster, a_cluster))
                            group = []

                    if len(group) > 0:
                        inter, corr, tot = evaluate_group(group, clustering_level)
                        n_interactions+=inter
                        n_correct += corr
                        n_total += tot
                        # print('tappendign group c: {} k {} a {}'.format(c, k_cluster, a_cluster))
                        group = []

        evals.append((n_interactions, n_correct, n_total))
        print("{} interactions, {} correct, {} total".format(n_interactions, n_correct, n_total))
    np.savetxt('perf_effort_unsupervised.csv', np.array(evals), delimiter=',')



    

if __name__ == '__main__':
    main()