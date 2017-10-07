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
class_list = genfromtxt('classes.csv', delimiter=',')

def main():
	reader = DataReader(images_directory, width, width, channels, class_list)
	Thread(target=lambda: start_server(reader)).start()

	x = SSLModel(width, width, channels, mb_size, len(class_list))

	for i in range(1000):

		lab = reader.minibatch_labeled(mb_size)

		if lab is None:
			time.sleep(1)
		else:
			print(x.train_step(reader.minibatch_unlabeled(mb_size)))


if __name__ == '__main__':
	main()