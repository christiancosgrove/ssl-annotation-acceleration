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
        #whether this image is currently in the test set
        self.test = False 

        self.url = None
import csv

class DataReader:
    def __init__(self, directory, width, height, channels, class_list, cache=True, load_filename=None):
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
                            urls[os.path.join('images', fname)] = row[1]
                        except: 
                            pass
            except Exception as e:
                pass

            for i, filename in enumerate(glob.glob('images/*')):
                info = ImageInfo(filename, len(self.class_list))
                #FOR DEBUGGING PURPOSES: set initial labels based on filenames
                if np.random.uniform() < 0.05:
                    for j, cname in enumerate(class_list):
                        if os.path.basename(filename).startswith(cname):
                            info.labels[j] = 1


                if urls.get(filename) is not None:
                    info.url = urls[filename]
                self.image_list.append(info)

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
            self.images[:] = np.array(np.stack([scipy.misc.imresize(scipy.ndimage.imread(v.name), (self.width, self.height)) for i,v in enumerate(self.image_list)]),dtype=np.float32)
            self.images[:] = self.images[:] / 127.5 - 1.0

    def save_image_list(self, filename):
        pickle.dump(self.image_list, open(filename, "wb"))

    def load_image_list(self, filename):
        self.image_list = pickle.load(open(filename, "wb"))

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
                    return None


            im = self.image_list[permutation[i]]
            if not im.autolabeled and not im.test:
                if positive:
                    cnum = np.argmax(im.labels)
                    if im.labels[cnum] > 0:
                        indices.append(permutation[i])
                        labels.append(cnum)
                else:
                    for cnum, v in np.ndenumerate(im.labels):
                        if v == -1:
                            indices.append(permutation[i])
                            labels.append(cnum)
            i+=1


        return np.array(self.images[indices]), np.array(labels)

    def label_image_positive(self, index, category):
        self.image_list[index].labels[category] = 1

        if np.random.uniform() < 0.5:
            self.image_list[index].test = True # randomly assign test label
    def label_image_negative(self, index, category):
        self.image_list[index].labels[category] = -1


    def get_labeling_batch(self, num_images):
        indices = []
        names = []

        permutation = np.random.permutation(len(self.image_list))
        i = 0
        while len(indices) < num_images:
            if i >= len(permutation):
                return None

            im = self.image_list[permutation[i]]
            cnum = np.argmax(im.labels)
            if im.labels[cnum] == 0:
                indices.append(permutation[i])
                if self.image_list[permutation[i]].url is not None:
                    names.append(self.image_list[permutation[i]].url)
                else:
                    names.append(self.image_list[permutation[i]].name)

            i+=1

        indices = np.array(indices)
        predictions = np.array([np.argmax(self.image_list[ind].prediction, axis=0) for ind in indices])
        return indices, names, predictions


    def autolabel(self, ssl_model, threshold): # where threshold is a confidence threshold, above which an image is automatically positive
        confidences = np.empty((0,len(self.class_list)))
        
        for i in range(len(self.image_list) // ssl_model.mb_size):
            confidences = np.append(confidences, ssl_model.predict(self.images[i * ssl_model.mb_size : (i + 1) * ssl_model.mb_size]), axis=0)

        count = 0

        correct_count = 0
        total_labeled = 0
        for i, conf in enumerate(confidences):
            self.image_list[i].prediction = conf

            m = np.argmax(conf, axis=0)
            if conf[m] > threshold and np.max(self.image_list[i].labels) == 0:
                if self.image_list[i].labels[m] == 0:
                    self.image_list[i].labels[m] = 1
                    self.image_list[i].autolabeled = True
                    count += 1

            c = np.argmax(self.image_list[i].labels)
            if self.image_list[i].labels[c] == 1:
                total_labeled+=1
                if m == c:
                    correct_count+=1

        print("{} images autolabeled.".format(count))
        print("{:.3f} training accuracy".format(correct_count / total_labeled))

    #returns a test accuracy on the current test set of the model
    def evaluate_model(self, ssl_model):
        test_indices = [i for i, im in enumerate(self.image_list) if im.test]

        # too few test indices!
        if len(test_indices) < ssl_model.mb_size:
            return None

        # in case test_indices is not the right length, include some wrap
        indices_modified = np.repeat(test_indices, 2)[:mb_size*int(np.ceil(len(test_indices) / ssl_model.mb_size))]
        confidences = np.empty((0,len(indices_modified)))

        for i in range(len(indices_modified) // ssl_model.mb_size):
            confidences = np.append(
                confidences,
                ssl_model.predict(self.images[
                    indices_modified[i * ssl_model.mb_size : (i + 1) * ssl_model.mb_size]]), axis=0)

        correct_count = 0
        total_labeled = 0

        for i in range(len(test_indices)):
            c_pred = np.argmax(confidences[i])
            c_test = np.argmax(self.image_list(test_indices[i]).labels)
            if c_pred == c_test:
                correct_count+=1
            total_labeled+=1

        return (correct_count, total_labeled)




    def start_new_test_period(self):
        for im in self.image_list:
            im.test = False
