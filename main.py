from data import DataReader
import matplotlib.pyplot as plt

from model import SSLModel

import time
from threading import Thread
from web import start_server
import numpy as np

mb_size = 64
images_directory = 'images'
width = 32
channels = 3

chunk_size = 1

from numpy import genfromtxt
class_list = [x.decode('ascii') for x in genfromtxt('classes.csv', delimiter=',', dtype=None)]

ITERATIONS = 100000

def main():
	reader = DataReader(images_directory, width, width, channels, class_list)

	model = SSLModel(width, width, channels, mb_size, len(class_list))
	Thread(target=lambda: start_server(reader, model)).start()

	for e in range(ITERATIONS):

		t = time.time()
		chunk_lab = reader.minibatch_labeled(mb_size, True)
		chunk_unl = reader.minibatch_unlabeled(mb_size)
		t = time.time()
		if chunk_lab is None:
			continue
		else:
			Y_neg = np.array([11] * mb_size, np.int64)

			dloss, gloss = model.train_step(chunk_unl, chunk_lab[0], chunk_lab[1], chunk_lab[0], chunk_lab[1])
			print('.', end='', flush=True)


			if e % 100 == 0:
				print(dloss, gloss)
				fake = model.sample_fake()[0]
				fake = fake * 0.5 + 0.5
				plt.imshow(fake)
				plt.savefig("generated.png")
				plt.close()

				reader.autolabel(model, 0.95, mb_size)


if __name__ == '__main__':
	main()