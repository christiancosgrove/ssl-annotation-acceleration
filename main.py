from data import DataReader
import matplotlib.pyplot as plt

from model import SSLModel

import time
from threading import Thread
from web import start_server




mb_size = 64
images_directory = 'images'
width = 32
channels = 3
num_classes = 10

def __main__():
	reader = DataReader(images_directory, width, width, channels, classes)
	Thread(target=lambda: start_server(reader)).start()

	x = SSLModel(width, width, channels, mb_size, classes)

	for i in range(1000):

		lab = reader.minibatch_labeled(mb_size)

		if lab is None:
			time.sleep(1)
		else:
			print(x.train_step(reader.minibatch_unlabeled(mb_size)))