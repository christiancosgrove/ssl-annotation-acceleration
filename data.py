from PIL import Image
import scipy.ndimage
import scipy.misc
import glob
import numpy as np
import matplotlib.pyplot as plt
import hashlib
import os
import base64

import pickle

class ImageInfo:
    def __init__(self, name, classes):
        self.name = name
        self.labels = np.array([0] * classes)

class DataReader:
    def __init__(self, directory, width, height, channels, classes, cache=True, load_filename=None):
        self.image_list = {}
        self.width = width
        self.height=height
        self.channels = channels
        self.classes = classes

        if load_filename is not None:
            self.load_image_list(load_filename)
        else:
            for i, filename in enumerate(glob.glob('images/*')):
                self.image_list[i] = ImageInfo(filename, classes)
        self.load('./cache',cache)

    def load(self, directory, cache=False): #loads images from directory; uses cache if available 

        h = hashlib.md5()
        for i, image in sorted(self.image_list.items(), key=lambda x: x[1].name):
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
            self.images[:] = np.array(np.stack([scipy.misc.imresize(scipy.ndimage.imread(self.image_list[i].name), (self.width, self.height)) for i in self.image_list]),dtype=np.float32)
            self.images[:] = self.images[:] / 127.5 - 1.0

    def save_image_list(self, filename):
        pickle.dump(self.image_list, open(filename, "wb"))

    def load_image_list(self, filename):
        self.image_list = pickle.load(open(filename, "wb"))

    def minibatch_unlabeled(self, mb_size):
        return self.images[np.random.randint(self.images.shape[0], size=mb_size),:,:,:]

    def minibatch_labeled(self, mb_size):
        indices = []
        labels = []

        permutation = np.random.permutation(len(self.image_list))
        i = 0
        while len(indices) < mb_size:
            if i >= len(permutation):
                return None

            im = self.image_list[permutation[i]]
            cnum = np.argmax(im.labels)
            if im.labels[cnum] > 0:
                indices.append(permutation[i])
                labels.append(cnum)

            i+=1

        return self.images[indices], np.array(labels)


