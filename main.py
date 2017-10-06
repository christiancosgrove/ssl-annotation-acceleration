from data import DataReader
import matplotlib.pyplot as plt

import time

x = DataReader('./images', 32,32, 3)
start = time.time()
ims = x.minibatch(10)

print(time.time() - start)
print(ims.shape)
plt.imshow(ims[0])
plt.show()