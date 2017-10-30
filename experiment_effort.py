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

    for clustering_level in range(levels):

        indices = np.random.permutation(len(reader.image_list))

        # randomly sample groups of images of size 32
        max_group_size = 32

        print("clustering level {}".format(clustering_level))

        #all possible cluster ranges
        #first two ranges must have a 
        sizes = []
        for i in range(CLUSTER_DEPTH):
            sizes.append(2**(i+1))

        n_interactions = 0
        n_correct = 0
        n_total = 0
        def evaluate_group(group, n_interactions, n_correct, n_total):

            class_preds = [0] * len(class_list)
            for ind in group:
                class_preds[reader.image_list[ind].ground_truth] += 1

            plurality_prediction = np.argmax(class_preds)
            for ind in group:
                if reader.image_list[ind].ground_truth == plurality_prediction:
                    n_correct+=1
                n_total+=1
            n_interactions +=1

            return n_interactions, n_correct, n_total

        for c in range(len(class_list)):
            for k_cluster in range(sizes[1]):
                for a_cluster in range(sizes[clustering_level - 2]):
                    i = 0
                    group = []
                    while i < len(indices):
                        if reader.image_list[indices[i]].clusters[0] == c:
                            if clustering_level == 0: # just clustering by class
                                group.append(indices[i])
                            else:
                                if (reader.image_list[indices[i]].clusters[1] == k_cluster):
                                    if clustering_level == 1:
                                        group.append(indices[i])
                                    else:
                                        if (reader.image_list[indices[i]].clusters[clustering_level] == a_cluster):
                                            group.append(indices[i])

                        i+=1

                        if len(group) >= max_group_size:
                            n_interactions, n_correct, n_total = evaluate_group(group, n_interactions, n_correct, n_total)
                            # print('appendign group c: {} k {} a {}'.format(c, k_cluster, a_cluster))
                            group = []

                    if len(group) > 0:
                        n_interactions, n_correct, n_total = evaluate_group(group, n_interactions, n_correct, n_total)
                        # print('tappendign group c: {} k {} a {}'.format(c, k_cluster, a_cluster))
                        group = []

                    if clustering_level < 2:
                        break
                if clustering_level < 1:
                    break


        print("{} interactions, {} correct, {} total".format(n_interactions, n_correct, n_total))



    

if __name__ == '__main__':
    main()