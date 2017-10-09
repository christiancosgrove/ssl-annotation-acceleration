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
        self.name = name
        self.labels = np.array([0] * classes)

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
            for i, filename in enumerate(glob.glob('images/*')):
                info = ImageInfo(filename, len(self.class_list))
                if os.path.basename(filename).startswith('i'):
                    info.labels[0] = 1
                if os.path.basename(filename).startswith('t'):
                    info.labels[1] = 1
                if os.path.basename(filename).startswith('s'):
                    info.labels[2] = 1
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

    def minibatch_labeled(self, mb_size, class_index):
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
            if im.labels[class_index] == 1:
                indices.append(permutation[i])
                labels.append(1.0)
            if im.labels[class_index] == -1:
                indices.append(permutation[i])
                labels.append(0.0)

            i+=1

        return np.array(self.images[indices]), np.array(labels)

    def label_image_positive(self, index, category):
        self.image_list[index].labels[category] = 1
    def label_image_negative(self, index, category):
        self.image_list[index].labels[category] = -1


    def get_labeling_batch(self, num_images, ssl_model):
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
                names.append(self.image_list[permutation[i]].name)

            i+=1

        indices = np.array(indices)
        predictions = np.argmax(ssl_model.predict(self.images[indices]), axis=1)
        return indices, names, predictions


    def autolabel(self, ssl_model, threshold, mb_size): # where threshold is a confidence threshold, above which an image is automatically positive
        confidences = np.empty((0,len(self.class_list)))
        
        for i in range(len(self.image_list) // mb_size):
            confidences = np.append(confidences, ssl_model.predict(self.images[i * mb_size : (i + 1) * mb_size]), axis=0)

        count = 0

        correct_count = 0
        total_labeled = 0
        for i, conf in enumerate(confidences):
            m = np.argmax(conf, axis=0)
            if conf[m] > threshold and np.max(self.image_list[i].labels) == 0:
                if self.image_list[i].labels[m] == 0:
                    self.image_list[i].labels[m] = 1
                    count += 1

            c = np.argmax(self.image_list[i].labels)
            if self.image_list[i].labels[c] == 1:
                total_labeled+=1
                if m == c:
                    correct_count+=1

        print("{} images autolabeled.".format(count))
        print("{:.3f} training accuracy".format(correct_count / total_labeled))

