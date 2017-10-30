from data import DataReader
import matplotlib.pyplot as plt

from model import SSLModel

import time
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--load', dest='LOAD', action='store_true')
parser.add_argument('--checkpoint_dir', dest='CHECKPOINT_DIR', nargs='?', const='checkpoints')
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

	os.makedirs(args.CHECKPOINT_DIR, exist_ok=True)

	model = SSLModel(width, width, channels, mb_size, len(class_list), args.CHECKPOINT_DIR, load=args.LOAD, use_generator=False)

	thresholds = np.arange(0.0,1.0,0.01);
	evals = []

	total_test_set = 10000
	# evals.append(("confidence threshold", "accuracy", "fraction of images above confidence threshold"))

	for t in thresholds:
		correct,total = reader.evaluate_model(model, threshold=t)
		print(t)
		if correct is not None and total is not None:
			if total > 0:
				evals.append((t, correct/total, total/total_test_set))

	np.savetxt('perf.csv', np.array(evals), delimiter=',')

if __name__ == '__main__':
	main()