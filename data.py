from PIL import Image
import scipy.ndimage
import scipy.misc
import glob
import numpy as np
import matplotlib.pyplot as plt
import hashlib
import os
import base64

class ImageInfo:
	def __init__(self, index):
		self.index = index


class DataReader:
	def __init__(self, directory, width, height, channels, cache=True):
		self.image_list = {}
		self.width = width
		self.height=height
		self.channels = channels
		for i, filename in enumerate(glob.glob('images/*')):
			self.image_list[filename] = ImageInfo(i)
		self.try_serialize('./cache',cache)

	def try_serialize(self, directory, cache=False): #tries to serialize 

		h = hashlib.md5()
		for fname in self.image_list:
			h.update(fname.encode('ascii','ignore'))

		name = base64.b16encode(h.digest()).decode('ascii')# unique, hashed name for the file
		name+= ".npz"
		name = os.path.join(directory,name)
		os.makedirs(directory,exist_ok=True)
		if os.path.isfile(name) and cache:
			self.images = np.memmap(name, dtype='float32', mode='r', shape=(len(self.image_list),self.width,self.height,self.channels))
		else:
			print("Converting images to dataset file...")
			self.images = np.memmap(name, dtype='float32', mode='w+', shape=(len(self.image_list),self.width,self.height,self.channels))
			self.images[:] = np.array(np.stack([scipy.misc.imresize(scipy.ndimage.imread(fname), (self.width, self.height)) for fname in self.image_list]),dtype=np.float32)
			self.images[:] = self.images[:] / 127.5 - 1.0


	def minibatch(self, mb_size):
		return self.images[np.random.randint(self.images.shape[0], size=mb_size),:,:,:]