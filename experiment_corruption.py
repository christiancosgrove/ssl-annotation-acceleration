from data import DataReader
import matplotlib.pyplot as plt

from model import SSLModel

import time
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--load', dest='LOAD', action='store_true')
# parser.add_argument('--checkpoint_dir', dest='CHECKPOINT_DIR', nargs='?', const='checkpoints')
args = parser.parse_args()

mb_size = 32
images_directory = 'images'
width = 32
channels = 3

chunk_size = 100

from numpy import genfromtxt
class_list = [x.decode('ascii') for x in genfromtxt('classes.csv', delimiter=',', dtype=None)]

import os

def main():
	reader = DataReader(images_directory, width, width, channels, class_list)
	evals = []
	model = SSLModel(width, width, channels, mb_size, len(class_list), "checkpoints_4000_unsupervised", load=True, use_generator=True)


	correct, total = reader.evaluate_model(model)
	base_performance = correct/total

	print("base performance: {}".format(base_performance))

	evals.append((0, correct / total / base_performance, correct/total))
	corruptions = [5,10,15,20,30,50]
	for corruption in corruptions:
		model.load('checkpoints_supervised_corr_{:02d}'.format(corruption))

		correct,total = reader.evaluate_model(model)
		evals.append((corruption, correct / total / base_performance, correct/total))

	np.savetxt('perf_corruption_supervised.csv', np.array(evals), delimiter=',')

if __name__ == '__main__':
	main()