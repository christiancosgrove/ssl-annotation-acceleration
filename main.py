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

chunk_size = 1

from numpy import genfromtxt
class_list = [x.decode('ascii') for x in genfromtxt('classes.csv', delimiter=',', dtype=None)]

def main():
	reader = DataReader(images_directory, width, width, channels, class_list)

	model = SSLModel(width, width, channels, mb_size, len(class_list))
	Thread(target=lambda: start_server(reader, model)).start()

	for e in range(1000):

		c = e % 3

		t = time.time()
		chunk_lab = reader.minibatch_labeled(chunk_size * mb_size, c)
		chunk_unl = reader.minibatch_unlabeled(chunk_size * mb_size)
		t = time.time()
		if chunk_lab is None:
			continue
		else:
			i = 0
			# for i in range(chunk_size):
			X_u = chunk_unl[i * mb_size : (i+1) * mb_size]
			X_l = chunk_lab[0][i * mb_size : (i+1) * mb_size]
			Y = chunk_lab[1][i * mb_size : (i+1) * mb_size]
			# if X_u.shape[0] != mb_size or X_l.shape[0] != mb_size or Y.shape[0] != mb_size:
			# 	continue
			dloss, gloss = model.train_step(X_u, X_l, Y, c)
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