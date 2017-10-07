import tensorflow as tf

class SSLModel:
	__init__(self, width, height, channels, mb_size):
		self.X = tf.Placeholder()