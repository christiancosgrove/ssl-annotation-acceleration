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

from numpy import genfromtxt
class_list = [x.decode('ascii') for x in genfromtxt('classes.csv', delimiter=',', dtype=None)]

def main():
	reader = DataReader(images_directory, width, width, channels, class_list)

	model = SSLModel(width, width, channels, mb_size, len(class_list))
	Thread(target=lambda: start_server(reader, model)).start()

	for i in range(1000):

		lab = reader.minibatch_labeled(mb_size)

		if lab is None:
			time.sleep(1)
		else:
			print(model.train_step(reader.minibatch_unlabeled(mb_size), lab[0], lab[1]))


if __name__ == '__main__':
	main()