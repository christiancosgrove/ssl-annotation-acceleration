import scipy.ndimage
import os
import glob

for i, filename in enumerate(glob.glob('images/*')):
	try:
		scipy.ndimage.imread(filename)
	except:
		print("deleting" + filename)
		os.remove(filename)