from data import DataReader
import matplotlib.pyplot as plt

from model import SSLModel

import time
from threading import Thread
from web_cluster import start_server
import numpy as np
import scipy.misc
import argparse
import os
from numpy import genfromtxt
parser = argparse.ArgumentParser()
parser.add_argument('--load', dest='LOAD', action='store_true')
parser.add_argument('--supervised', dest='SUPERVISED', action='store_true')
parser.add_argument('--web', dest='WEB', action='store_true')
parser.add_argument('--checkpoint_dir', dest='CHECKPOINT_DIR', default='checkpoints')
parser.add_argument('--corruption', dest='CORRUPTION', default=0.0, type=float)
parser.add_argument('--no_train', dest='NO_TRAIN', action='store_true')
args = parser.parse_args()

mb_size = 32
images_directory = 'images'
width = 32
channels = 3

chunk_size = 100

class_list = [x.decode('ascii') for x in genfromtxt('classes.csv', delimiter=',', dtype=None)]

ITERATIONS = 2000

def main():
    reader = DataReader(images_directory, width, width, channels, class_list, corruption=args.CORRUPTION)

    # reader.save_image_list('image_list.pkl')

    if args.WEB:
        Thread(target=lambda: start_server(reader)).start()
    os.makedirs(args.CHECKPOINT_DIR, exist_ok=True)

    model = SSLModel(width, width, channels, mb_size, len(class_list), args.CHECKPOINT_DIR, load=args.LOAD, use_generator=not args.SUPERVISED)
    reader.autolabel(model, 1.1)

    for e in range(ITERATIONS):
        
        t = time.time()
        chunk_lab = reader.minibatch_labeled(mb_size * chunk_size, True)
        chunk_neg = reader.minibatch_labeled(mb_size * chunk_size, False)
        chunk_unl = reader.minibatch_unlabeled(mb_size * chunk_size)
        t = time.time()
        if chunk_lab is None:
            continue
        else:

            dloss = 0
            gloss = 0
            if chunk_neg is not None and not args.NO_TRAIN:
                for i in range(chunk_size):
                    X_mb = chunk_unl[i * mb_size : (i+1) * mb_size]
                    if X_mb.shape[0] < mb_size:
                        continue
                    X_lab_mb = chunk_lab[0][i * mb_size : (i+1) * mb_size]
                    Y_mb = chunk_lab[1][i * mb_size : (i+1) * mb_size]
                    X_neg_mb = chunk_neg[0][i * mb_size : (i+1) * mb_size] if chunk_neg is not None else X_lab_mb
                    Y_neg_mb = np.squeeze(chunk_neg[1][i * mb_size : (i+1) * mb_size]) if chunk_neg is not None else np.array([11] * mb_size, np.int64)
                    if X_neg_mb.shape[0] > 0 and X_neg_mb.shape[0] < mb_size:
                        shape = int(np.ceil(X_neg_mb.shape[0] / mb_size))
                        X_neg_mb = np.tile(X_neg_mb, (shape, 1, 1, 1))
                        Y_neg_mb = np.tile(Y_neg_mb, (shape, 1, 1, 1))
                    # if chunk_neg is not None and chunk_neg.shape[0] == mb_size:
                    dloss, gloss = model.train_step(X_mb, X_lab_mb, Y_mb, X_neg_mb, Y_neg_mb)
                    print('.', end='', flush=True)

            correct_count, total_labeled = reader.evaluate_model(model)

            if correct_count is not None and total_labeled is not None:
                if total_labeled != 0:
                    percent = int(correct_count / total_labeled * 100)
                else:
                    percent = 0
                print("epoch {}, {} correct, {} total from test set, {}% correct".format(e, correct_count, total_labeled, percent));
            else:
                print("Could not evaluate model")
            if e % 5 == 0:
                print(dloss, gloss)
                fake = model.sample_fake()
                fake = fake * 0.5 + 0.5
                fake = np.reshape(fake, (mb_size * width, width, channels))
                scipy.misc.imsave('generated.png', fake)

                reader.autolabel(model, 1.1)
                model.save()

if __name__ == '__main__':
    main()