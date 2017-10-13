from data import DataReader
import matplotlib.pyplot as plt

from model import SSLModel

import time
from threading import Thread
from web import start_server
import numpy as np
import scipy.misc

mb_size = 32
images_directory = 'images'
width = 32
channels = 3

chunk_size = 100

from numpy import genfromtxt
class_list = [x.decode('ascii') for x in genfromtxt('classes.csv', delimiter=',', dtype=None)]

ITERATIONS = 100000

def main():
	reader = DataReader(images_directory, width, width, channels, class_list)

	Thread(target=lambda: start_server(reader)).start()
	model = SSLModel(width, width, channels, mb_size, len(class_list))

	for e in range(ITERATIONS):

		t = time.time()
		chunk_lab = reader.minibatch_labeled(mb_size * chunk_size, True)
		chunk_neg = reader.minibatch_labeled(mb_size * chunk_size, False)
		chunk_unl = reader.minibatch_unlabeled(mb_size * chunk_size)
		t = time.time()
		if chunk_lab is None:
			continue
		else:

			for i in range(chunk_size):
				X_mb = chunk_unl[i * mb_size : (i+1) * mb_size]
				X_lab_mb = chunk_lab[0][i * mb_size : (i+1) * mb_size]
				Y_mb = chunk_lab[1][i * mb_size : (i+1) * mb_size]
				Y_neg = chunk_neg if chunk_neg is not None else np.array([11] * mb_size, np.int64)

				dloss, gloss = model.train_step(X_mb, X_lab_mb, Y_mb, X_lab_mb, Y_neg)
				print('.', end='', flush=True)



			correct_count, total_labeled = reader.evaluate_model(model)

			print("{} correct, {} total from test set, {}% correct".format(correct_count, total_labeled, int(correct_count / total_labeled * 100)));

			if e % 5 == 0:
				print(dloss, gloss)
				fake = model.sample_fake()[0]
				fake = fake * 0.5 + 0.5
				scipy.misc.imsave('generated.png', fake)

				reader.autolabel(model, 0.95)


if __name__ == '__main__':
	main()