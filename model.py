import tensorflow as tf
import numpy as np
import os

class SSLModel:
    def __init__(self, width, height, channels, mb_size, classes, checkpoint_dir, z_dim=100, learning_rate=3e-4, beta=0.5, load=False, use_generator=True):
        self.width = width
        self.height = height
        self.channels = channels
        self.mb_size = mb_size
        self.classes = classes
        self.z_dim = z_dim
        self.learning_rate = learning_rate
        self.beta = beta
        self.use_generator = use_generator

        self.X = tf.placeholder(tf.float32, shape=[mb_size, width, height, channels])
        self.X_lab = tf.placeholder(tf.float32, shape=[mb_size, width, height, channels])
        self.X_lab_neg = tf.placeholder(tf.float32, shape=[mb_size, width, height, channels])
        # self.Y = tf.placeholder(tf.float32, shape=[mb_size])
        self.Y = tf.placeholder(tf.int64, shape=[mb_size])
        self.Y_neg = tf.placeholder(tf.int64, shape=[mb_size])
        self.z = tf.placeholder(tf.float32, shape=[mb_size, self.z_dim])
        self.training_now = tf.placeholder(tf.bool)


        self.global_step = 0

        self.checkpoint_dir = checkpoint_dir
        self.build(load) 

    def build(self, load):
        
        self.X_fake = self.G(self.z)
        self.D_real, self.D_real_feat = self.D(self.X)

        self.class_probabilities = tf.nn.softmax(self.D_real)

        self.D_real_lab, _     = self.D(self.X_lab, True)
        self.D_real_lab_neg, _ = self.D(self.X_lab_neg, True)
        self.D_fake, self.D_fake_feat = self.D(self.X_fake, True)

        l_enc = tf.reduce_logsumexp(self.D_real, axis=1)
        l_gen = tf.reduce_logsumexp(self.D_fake, axis=1)

        self.D_loss_unl = (-tf.reduce_mean(l_enc) + tf.reduce_mean(tf.nn.softplus(l_enc)) + tf.reduce_mean(tf.nn.softplus(l_gen)))
        # self.D_loss_lab = []

        self.D_loss_lab = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.D_real_lab, labels=self.Y))
        
        #negative loss
        negative_logits = tf.gather(self.D_real_lab_neg, self.Y_neg, validate_indices=True, axis=-1)
        self.D_loss_lab+= tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=negative_logits, labels=tf.zeros_like(negative_logits)))


        # for i in range(self.classes):
        #     class_logit = tf.squeeze(tf.slice(self.D_real_lab, [0,i],[-1,1]))
        #     self.D_loss_lab.append(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=class_logit, labels=self.Y)))
        # self.D_loss = [self.D_loss_unl + x for x in self.D_loss_lab]
        self.D_loss = self.D_loss_lab
        if self.use_generator:
            self.D_loss += self.D_loss_unl
        self.G_loss = tf.reduce_mean(tf.square(tf.reduce_mean(self.D_real_feat, axis=0)-tf.reduce_mean(self.D_fake_feat, axis=0)))

        #pull-away term - increases entropy of generated images (measured by discriminator features)
        #first applied to GANs in Dai et al. 2017 https://arxiv.org/abs/1705.09783
        feat_norm = self.D_fake_feat / tf.norm(self.D_fake_feat, axis=1, keep_dims=True)
        G_pt = tf.tensordot(feat_norm, feat_norm, axes=[[1],[1]])
        self.G_loss += tf.reduce_mean(G_pt)

        theta_G = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='G_')
        theta_D = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='D_')

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta)
            # opt = tf.contrib.opt.MovingAverageOptimizer(opt)
            # self.D_solver = [(opt.minimize(x, var_list=theta_D)) for x in self.D_loss]
            self.D_solver = opt.minimize(self.D_loss, var_list=theta_D)
            self.G_solver = (opt.minimize(self.G_loss, var_list=theta_G))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = False
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        if load:
            self.saver.restore(self.sess, tf.train.latest_checkpoint(checkpoint_dir=self.checkpoint_dir,latest_filename='checkpoint'))

    def G(self, z):
        with tf.variable_scope('G_'):
            h = z
            h = bn_dense(z, 4*4*512, self.training_now, bias=False, nonlinearity=tf.nn.relu)
            h = tf.reshape(h, (self.mb_size, 4, 4, 512))
            h = bn_convlayer_t(h, 5, 2, 256, self.training_now, bias=False, nonlinearity=tf.nn.relu)
            h = bn_convlayer_t(h, 5, 2, 128, self.training_now, bias=False, nonlinearity=tf.nn.relu)
            h = tf.layers.conv2d_transpose(inputs=h, filters=3, kernel_size=[5,5], padding='same', activation=tf.nn.tanh, strides=(2,2), use_bias=True)
            return h

    def D(self, X, reuse=False):
        with tf.variable_scope('D_', reuse=reuse):
            h_X = dropout_layer(X, 0.2, self.training_now)
            h_X = convlayer(h_X, 3, 1, 96)
            h_X = convlayer(h_X, 3, 1, 96)
            h_X = convlayer(h_X, 3, 2, 96)
            h_X = dropout_layer(h_X, 0.5, self.training_now)
            h_X = convlayer(h_X, 3, 1, 192)
            h_X = convlayer(h_X, 3, 1, 192)
            h_X = convlayer(h_X, 3, 2, 192)
            h_X = dropout_layer(h_X, 0.5, self.training_now)
            h_X = convlayer(h_X, 3, 1, 192)
            h_X = convlayer(h_X, 1, 1, 192)
            h_X = convlayer(h_X, 1, 1, 192)
            h_X = global_pool(h_X, 8)

            h = tf.layers.dense(h_X, self.classes)
            return h, h_X

    def train_step(self, X_mb, X_lab_mb, Y_mb, X_lab_neg_mb, Y_neg_mb):
        z_mb = self.sample_z()

        _, D_loss_curr = self.sess.run(
            [self.D_solver, self.D_loss], feed_dict=
                {
                    self.X: X_mb,
                    self.z: z_mb,
                    self.X_lab: X_lab_mb,
                    self.Y: Y_mb,
                    self.X_lab_neg: X_lab_neg_mb,
                    self.Y_neg: Y_neg_mb,
                    self.training_now:True
                })

        if self.use_generator:
            _, G_loss_curr = self.sess.run(
                [self.G_solver, self.G_loss], feed_dict={self.X: X_mb, self.z: z_mb, self.X_lab: X_lab_mb, self.Y: Y_mb, self.training_now:True}
            )

        self.global_step += 1

        return (D_loss_curr, G_loss_curr)

    def sample_z(self):
        return np.random.uniform(-1., 1., size=[self.mb_size, self.z_dim])

    def predict(self, X): #get class probabilities for a minibatch of images
        return self.sess.run(self.class_probabilities, feed_dict={self.X: X, self.training_now:False})

    def sample_fake(self):
        return self.sess.run(self.X_fake, feed_dict={self.z: self.sample_z(), self.training_now:False})

    def save(self):
        self.saver.save(sess=self.sess, save_path=os.path.join(self.checkpoint_dir,'checkpoint'), global_step=self.global_step)

def lrelu(x, leak=0.2, name="lrelu"):
     with tf.variable_scope(name):
         f1 = 0.5 * (1 + leak)
         f2 = 0.5 * (1 - leak)
         return f1 * x + f2 * tf.abs(x)
def convlayer(layer, filter_size, stride, filters, bias=True, nonlinearity=lrelu):
    return tf.layers.conv2d(inputs=layer, filters=filters, kernel_size=[filter_size,filter_size], padding='same', activation=nonlinearity, strides=(stride,stride), use_bias=bias)
def convlayer_t(layer, filter_size, stride, filters, bias=True, nonlinearity=lrelu):
    return tf.layers.conv2d_transpose(inputs=layer, filters=filters, kernel_size=[filter_size,filter_size], padding='same', activation=nonlinearity, strides=(stride,stride), use_bias=bias)

def bn_convlayer(layer, filter_size, stride, filters, trnow, bias=True, nonlinearity=lrelu):
    return tf.contrib.layers.batch_norm(tf.layers.conv2d(inputs=layer, filters=filters, kernel_size=[filter_size,filter_size], padding='same', strides=(stride,stride), use_bias=bias), is_training=trnow, fused=True, activation_fn=nonlinearity)
def bn_convlayer_t(layer, filter_size, stride, filters, trnow, bias=True, nonlinearity=lrelu):
    return tf.contrib.layers.batch_norm(tf.layers.conv2d_transpose(inputs=layer, filters=filters, kernel_size=[filter_size,filter_size], padding='same', strides=(stride,stride), use_bias=bias), is_training=trnow, fused=True, activation_fn=nonlinearity)

def dense(layer, units, bias=True, nonlinearity=lrelu):
    return tf.layers.dense(inputs=layer, units=units, activation=nonlinearity, use_bias=bias)
def bn_dense(layer, units, trnow, bias=True, nonlinearity=lrelu):
    return tf.contrib.layers.batch_norm(tf.layers.dense(inputs=layer, units=units, use_bias=bias), is_training=trnow, fused=True, activation_fn=nonlinearity)

def dropout_layer(layer, rate, trnow):
    return tf.layers.dropout(inputs=layer, rate=rate, training=trnow)

def global_pool(layer, old_width):
    return tf.squeeze(tf.layers.average_pooling2d(layer, [old_width, old_width], [old_width, old_width]))
