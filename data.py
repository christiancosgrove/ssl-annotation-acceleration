from PIL import Image
import scipy.ndimage
import scipy.misc
import glob
import numpy as np
import matplotlib.pyplot as plt
import hashlib
import os
import base64
from model import SSLModel
import pickle
import csv
import re

from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch

from hierarchical_birch import fit_predict_hierarchical_birch

CLUSTER_DEPTH = 8

class ImageInfo:
    def __init__(self, name, classes):
        #filename
        self.name = name

        #current label for this image
        #1 designates positive label (image belongs to class)
        #-1 designates negative label (image does not belong to class)
        #0 designates unknown
        self.labels = np.array([0] * classes)

        #whether the ground-truth labels contained in self.labels were assigned by the classifier
        #if so, do not use it to train the classifier, so as to avoid bias
        self.autolabeled = False

        #current classifier confidences for this image
        self.prediction = np.array([0] * classes)

        self.clusters = None

        #whether this image is currently in the test set
        self.test = False 

        # FOR EVALUATION PURPOSES (used when evaluating current technique on CIFAR-10)
        #the ground-truth label of the image
        self.ground_truth = None

        self.url = None

class DataReader:
    def __init__(self, directory, width, height, channels, class_list, cache=True, load_filename=None, evaluating=False, corruption=0.0):
        self.image_list = []
        self.width = width
        self.height=height
        self.channels = channels
        self.class_list = class_list

        if load_filename is not None:
            self.load_image_list(load_filename)
        else:
            urls = {}
            try:
                with open('urls.csv', 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        try:
                            fname, url = row
                            urls[os.path.join(directory, fname)] = row[1]
                        except: 
                            pass
            except Exception as e:
                pass

            for i, filename in enumerate(glob.glob(directory +'/*')):
                info = ImageInfo(filename, len(self.class_list))


                for j, cname in enumerate(class_list):
                    if re.match('\d+_' + cname, os.path.basename(filename)) is not None:
                        info.ground_truth = j
                        if re.match(r'.*test.*', filename) is not None:
                            info.labels[info.ground_truth] = 1
                            info.test = True

                if urls.get(filename) is not None:
                    info.url = urls[filename]
                self.image_list.append(info)

            np.random.seed(0)
            indices = np.random.permutation(len(self.image_list))
            i = 0
            while i < 50:#4000
                if not self.image_list[indices[i]].test:
                    if np.random.uniform() < corruption:
                        self.image_list[indices[i]].labels[np.random.randint(len(class_list))] = 1
                    else:
                        self.image_list[indices[i]].labels[self.image_list[indices[i]].ground_truth] = 1
                i += 1

                    # for j, cname in enumerate(class_list):

                        # if os.path.basename(filename).startswith(cname):
                        #     info.labels[j] = 1

                        #     if np.random.uniform() < 0.5:
                        #         info.test = True



        self.load('./cache',cache)

    def load(self, directory, cache=False): #loads images from directory; uses cache if available 

        h = hashlib.md5()
        for i, image in sorted(list(enumerate(self.image_list)), key=lambda x: x[1].name):
            h.update(image.name.encode('ascii','ignore'))

        name = base64.b16encode(h.digest()).decode('ascii')# unique, hashed name for the file
        name+= ".npz"
        name = os.path.join(directory,name)
        os.makedirs(directory,exist_ok=True)
        if os.path.isfile(name) and cache:
            self.images = np.memmap(name, dtype='float32', mode='r', shape=(len(self.image_list), self.width, self.height, self.channels))
        else:
            print("Converting images to dataset file...")
            self.images = np.memmap(name, dtype='float32', mode='w+', shape=(len(self.image_list), self.width, self.height, self.channels))
            
            percent = 0

            for i,v in enumerate(self.image_list):
                im = scipy.ndimage.imread(v.name)

                if im.shape[0] != self.width or im.shape[1] != self.height:
                    if im.shape[0] > im.shape[1]:
                        edge = (im.shape[0] - im.shape[1]) // 2
                        im = im[edge:-edge,:,:]
                    elif im.shape[1] > im.shape[0]:
                        edge = (im.shape[1] - im.shape[0]) // 2
                        im = im[:, edge:-edge, :]
                    im = scipy.misc.imresize(im, (self.width, self.height)) 
                self.images[i] = im / 127.5 - 1.0
                
                if 100 * i // len(self.image_list) != percent:
                    percent = 100 * i // len(self.image_list)
                    print("{}%".format(percent))


    def save_image_list(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.image_list, file)

    def load_image_list(self, filename):
        with open(filename, 'rb') as file:
            self.image_list = pickle.load(file)

    def minibatch_unlabeled(self, mb_size):
        permutation = np.random.permutation(len(self.image_list))
        return np.array(self.images[permutation[:mb_size]])

    #get positive or negative labeled minibatch
    def minibatch_labeled(self, mb_size, positive):
        indices = []
        labels = []

        permutation = np.random.permutation(len(self.image_list))
        i = 0
        while len(indices) < mb_size:
            if i >= len(permutation):
                if len(indices) > 0:
                    permutation = np.random.permutation(len(self.image_list))
                    i = 0
                else:
                    if not positive:
                        return self.minibatch_labeled_negative_from_positive(mb_size)
                    else:
                        return None


            im = self.image_list[permutation[i]]
            if not im.autolabeled and not im.test:
                if positive:
                    cnum = np.argmax(im.labels)
                    if im.labels[cnum] == 1:
                        indices.append(permutation[i])
                        labels.append(cnum)
                else:
                    for cnum, v in np.ndenumerate(im.labels):
                        if v == -1:
                            indices.append(permutation[i])
                            labels.append(cnum)
            i+=1


        return np.array(self.images[indices]), np.array(labels)

    def minibatch_labeled_negative_from_positive(self, mb_size):
        indices = []
        labels = []

        permutation = np.random.permutation(len(self.image_list))
        i = 0
        while len(indices) < mb_size:
            if i >= len(permutation):
                if len(indices) > 0:
                    permutation = np.random.permutation(len(self.image_list))
                    i = 0
                else:
                    return None


            im = self.image_list[permutation[i]]
            if not im.autolabeled and not im.test:
                cnum = np.argmax(im.labels)
                if im.labels[cnum] == 1:
                    neg_class = np.random.randint(len(im.labels))
                    while neg_class == cnum:
                        neg_class = np.random.randint(len(im.labels))

                    indices.append(permutation[i])
                    labels.append(neg_class)
            i+=1


        return np.array(self.images[indices]), np.array(labels)


    def label_image_positive(self, index, category):
        self.image_list[index].labels[category] = 1

        if np.random.uniform() < 0.5:
            self.image_list[index].test = True # randomly assign test label
    def label_image_negative(self, index, category):
        self.image_list[index].labels[category] = -1


    def append_to_names(self, index, names):
        if self.image_list[index].url is not None:
            names.append(self.image_list[index].url)
        else:
            names.append(self.image_list[index].name)
    def get_labeling_batch(self, num_images):
        indices = []
        names = []

        permutation = np.random.permutation(len(self.image_list))
        i = 0
        while len(indices) < num_images:
            if i >= len(permutation):
                return None, None, None

            im = self.image_list[permutation[i]]
            cnum = np.argmax(im.labels)
            if im.labels[cnum] == 0:
                indices.append(permutation[i])
                self.append_to_names(permutation[i], names)

            i+=1

        indices = np.array(indices)
        predictions = np.array([np.argmax(self.image_list[ind].prediction, axis=0) for ind in indices])
        return indices, names, predictions

    def get_labeling_batch_clustered(self, num_images, num_images_groundtruth):
        # in order to get a labeling batch, we must have called autolabel at least once - assigns classifier-derived clustering to each image in dataset
        if self.image_list[0].clusters is None: 
            return None, None, None, None, None
        #select a random class from which to select images
        c = np.random.randint(len(self.class_list))

        depth = 1
        selected_clusters = []
        candidates = [(i, im) for (i, im) in enumerate(self.image_list) if im.clusters is not None and im.labels is not None and im.clusters[0] == c and np.argmax(im.prediction) == c]
        while True:
            if self.image_list[0].clusters is None or depth >= len(self.image_list[0].clusters):
                break
            selected_cluster = np.random.randint(1)
            next_candidates = [(i, im) for (i, im) in candidates if im.clusters[depth] == selected_cluster]

            if len(next_candidates) < num_images:
                break
            candidates = next_candidates
            depth += 1
        candidates = candidates[:num_images]
        indices = np.array([i for (i, im) in candidates], dtype=np.int64)
        predictions = np.array([c] * len(candidates), dtype=np.int64)
        names = [im.name for (i, im) in candidates]
        clusters = np.array([im.clusters for (i, im) in candidates], dtype=np.int64)
        if clusters.shape[0] == 0:
            return None, None, None, None, None

        if num_images_groundtruth > 0:
            # get ground-truth images - used to evaluate performance of annotators
            # 50% are positive, 50% are negative

            num_positive_groundtruth = num_images_groundtruth // 2
            num_negative_groundtruth = num_images_groundtruth - num_positive_groundtruth

            candidates_positive = [(i, im) for (i, im) in enumerate(self.image_list) if im.clusters is not None and im.labels[c] == 1]
            candidates_positive = [candidates_positive[i] for i in np.random.permutation(min(num_positive_groundtruth, len(candidates_positive)))]
            indices_positive = np.array([i for (i, im) in candidates_positive], dtype=np.int64)
            predictions_positive = np.array([c] * len(candidates_positive), dtype=np.int64)
            names_positive = [im.name for (i, im) in candidates_positive]
            clusters_positive = np.array([im.clusters for (i, im) in candidates_positive], dtype=np.int64)

            indices_negative = np.empty(num_negative_groundtruth, dtype=np.int64)
            predictions_negative = np.empty(num_negative_groundtruth, dtype=np.int64)
            names_negative = []
            clusters_negative = np.empty((num_negative_groundtruth, clusters_positive.shape[1]), dtype=np.int64)
            for i in range(num_negative_groundtruth):
                neg_index = np.random.randint(len(self.image_list))
                while True:
                    c_neg = np.argmax(self.image_list[neg_index].labels)
                    if self.image_list[neg_index].labels[c_neg] == 1 and c_neg != c:
                        break
                    neg_index = np.random.randint(len(self.image_list))
                if self.image_list[neg_index].clusters is None:
                    break
                indices_negative[i] = neg_index
                predictions_negative[i] = c
                names_negative.append(self.image_list[neg_index].name)
                clusters_negative[i, :] = self.image_list[neg_index].clusters

            positives = np.array([0] * len(indices) + [1] * len(indices_positive) + [-1] * len(indices_negative), dtype=np.int64)
            indices = np.concatenate((indices, indices_positive, indices_negative))
            predictions = np.concatenate((predictions, predictions_positive, predictions_negative))
            names = names + names_positive + names_negative
            clusters = np.concatenate((clusters, clusters_positive, clusters_negative))

        return indices, names, predictions, clusters, positives


    def get_labeling_batch_groundtruth(self, num_images):
        indices = []
        names = []
        positives = []
        predictions = []

        permutation = np.random.permutation(len(self.image_list))
        i = 0
        while len(indices) < num_images:
            if i >= len(permutation):
                return np.array([], dtype=np.int64), [], np.array([], dtype=np.int64), np.array([], dtype=np.int64)

            im = self.image_list[permutation[i]]
            cnum = np.argmax(im.labels)

            if im.labels[cnum] == 1:
                indices.append(permutation[i])
                self.append_to_names(permutation[i], names)
                #half will be negative ground truth labels
                if np.random.uniform() < 0.5:
                    neg_class = np.random.randint(len(im.labels))
                    while neg_class == cnum:
                        neg_class = np.random.randint(len(im.labels))
                    predictions.append(neg_class)
                    positives.append(-1)
                else:
                    predictions.append(cnum)
                    positives.append(1)
            # else:
            #     perm = np.random.permutation(im.labels.shape[0])
            #     cnum = perm[np.argmin(im.labels[perm])]
            #     if im.labels[cnum] == -1:
            #         indices.append(permutation[i])
            #         predictions.append(cnum)
            #         positives.append(-1)
            #         append_to_names(self, permutation[i], names)
            i+=1

        indices = np.array(indices)
        predictions = np.array(predictions)
        positives = np.array(positives)
        return indices, names, predictions, positives

    def autolabel(self, ssl_model, threshold, use_clustering=True): # where threshold is a confidence threshold, above which an image is automatically positive
        confidences = np.empty((len(self.image_list),len(self.class_list)))
        features = None

        print("autolabeling")
        for i in range(len(self.image_list) // ssl_model.mb_size + 1):

            #get features and class confidences for every image in training set
            mb_features, mb_confidences = ssl_model.features_predict(self.images[i * ssl_model.mb_size : (i + 1) * ssl_model.mb_size])

            confidences[i * ssl_model.mb_size : (i + 1) * ssl_model.mb_size] = mb_confidences
            
            if features is None and use_clustering:
                features = np.empty((len(self.image_list),mb_features.shape[1]))
            features[i * ssl_model.mb_size : (i + 1) * ssl_model.mb_size] = mb_features
        print("got confidences")

        if features is not None:
            brc = Birch(branching_factor=2, n_clusters=None, compute_labels=True)
            print(brc.fit_predict(features.astype(np.float64)))
            shortest_clusters_length = -1
            for c in range(len(self.class_list)):
                group = [x for x in enumerate(features) if np.argmax(confidences[x[0]]) == c]
                clusters = fit_predict_hierarchical_birch(np.array([f[1] for f in group]))
                for i, x in enumerate(group):
                    self.image_list[x[0]].clusters = np.concatenate([[c], clusters[i]])
                if shortest_clusters_length == -1 or clusters.shape[1] < shortest_clusters_length:
                    shortest_clusters_length = clusters.shape[1]
            # truncate clusters to shortest clusters length across all classes 
            for im in self.image_list:
                im.clusters.resize(shortest_clusters_length)
        print(self.image_list[0].clusters)
        # if features is not None:
        #     n_clusters=50 #number of k-means clusters per class
        #     labels = MiniBatchKMeans(n_clusters=n_clusters, max_iter=10).fit_predict(features.astype(np.float64))
        #     labels = np.c_[np.arange(len(labels)),np.array(labels)]
        #     print('clustering class ')
        #     for c in range(len(self.class_list)):
        #         print('{} '.format(c), end='', flush=True)
        #         for l_index in range(n_clusters):

        #             group = np.array([x for x in labels if x[1] == l_index and np.argmax(confidences[x[0]])==c])
        #             if group.shape[0] <= 1:
        #                 continue
        #             # perform hierarchical agglomerative clustering on each k-means cluster
        #             # this agglomerative clustering implementation is memory hungry, thus we must split up the training set

        #             agg_labels = []

        #             n_agg_clusters = min(2**(CLUSTER_DEPTH), group.shape[0]) #make sure we don't use too many clusters - clamp to number of samples
        #             agg = AgglomerativeClustering(n_clusters=n_agg_clusters)
        #             for i in range(CLUSTER_DEPTH):
        #                 n_agg_clusters = min(2**(i+1), group.shape[0])
        #                 agg.set_params(n_clusters=n_agg_clusters)
        #                 agg_labels.append(agg.fit_predict(features[group[:,0]]))
        #             agg_labels = np.array(agg_labels)
        #             for i, x in enumerate(group):
        #                 self.image_list[x[0]].clusters = np.concatenate([[c, x[1]], agg_labels[:,i]])
            # print('')
        count = 0

        correct_count = 0
        total_labeled = 0
        for i, conf in enumerate(confidences):
            self.image_list[i].prediction = conf

            m = np.argmax(conf, axis=0)
            c = np.argmax(self.image_list[i].labels)
            if conf[m] > threshold and self.image_list[i].labels[c] == 0:
                if self.image_list[i].labels[m] == 0:
                    self.image_list[i].labels[m] = 1
                    self.image_list[i].autolabeled = True
                    count += 1
            if self.image_list[i].labels[c] == 1:
                total_labeled+=1
                if m == c:
                    correct_count+=1

        print("{} images autolabeled.".format(count))
        print("{:.3f} training accuracy".format(correct_count / total_labeled))

    #returns a test accuracy on the current test set of the model
    #threshold is the confidence above which accuracy is evaluated
    def evaluate_model(self, ssl_model, threshold=0.0):
        test_indices = [i for i, im in enumerate(self.image_list) if im.test]

        # too few test indices!
        if len(test_indices) < ssl_model.mb_size:
            return None, None

        # in case test_indices is not the right length, include some wrap
        indices_modified = np.tile(test_indices, 2)[:ssl_model.mb_size*int(np.ceil(len(test_indices) / ssl_model.mb_size))]
        confidences = np.empty((len(indices_modified),len(self.class_list)))

        for i in range(len(indices_modified) // ssl_model.mb_size):
            confidences[i * ssl_model.mb_size : (i + 1) * ssl_model.mb_size] = ssl_model.predict(self.images[
                    indices_modified[i * ssl_model.mb_size : (i + 1) * ssl_model.mb_size]])

        correct_count = 0
        total_labeled = 0

        for i in range(len(test_indices)):
            # print(confidences[i])
            c_pred = np.argmax(confidences[i], axis=0)

            # print(self.image_list[test_indices[i]].labels)
            c_test = np.argmax(self.image_list[test_indices[i]].labels, axis=0)
            if self.image_list[test_indices[i]].labels[c_test] != 1: # make sure we are evaluating on positive images only
                continue

            if confidences[i][c_pred] > threshold:
                if c_pred == c_test:
                    correct_count+=1
                total_labeled+=1

        return (correct_count, total_labeled)




    def start_new_test_period(self):
        for im in self.image_list:
            im.test = False
