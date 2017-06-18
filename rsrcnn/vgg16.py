# imports
import tensorflow as tf
from scipy import misc
import numpy as np
from imagenet_classes import class_names

class vgg16:
    def __init__(self, imgs, weights=None, sess=None):
        self.imgs = imgs
        self.conv = {}
        self.pool = {}
        self.fc   = {}
        self.probs = self.encoder(self.imgs, 'encoder')
        if weights is not None and sess is not None:
            self.load_weights(weights, sess)

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())

        for i, k in enumerate(keys):
            if(k=='conv1_1_W'):
                new_weights = np.zeros((3, 3, 4, 64))
                new_weights[:,:,3] = np.zeros((1, 64))
                print(i, k, np.shape(weights[k]))
                sess.run(self.parameters[i].assign(new_weights))
                continue
            print(i, k, np.shape(weights[k]))
            sess.run(self.parameters[i].assign(weights[k]))

    def conv2d(self, input, filter_shape, strides = (1,1,1,1), activation = tf.nn.relu, pad = "SAME", name = None, stddev=1e-1):
        with tf.variable_scope(name) as scope:
            kernel = tf.Variable(tf.truncated_normal(filter_shape, dtype=tf.float32,
                                                     stddev=stddev), name='weights')
            conv = tf.nn.conv2d(input, kernel, strides, padding=pad)
            biases = tf.Variable(tf.constant(0.0, shape=[filter_shape[3]], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)

            self.conv[name] = tf.nn.relu(out, name=scope.name)
            self.parameters += [kernel, biases]
            return self.conv[name]

    def max_pool(self, input, ksize=[1,2,2,1], strides = [1,2,2,1], padding = "SAME", name = None):
        with tf.variable_scope(name) as scope:
            res = tf.nn.max_pool(value=input, ksize=ksize, strides=strides, padding=padding)
        self.pool[name] = res
        return res

    def preprocess(self, imgs, name=None):
        with tf.variable_scope(name) as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        return imgs - mean

    def fc_layer(self, input, shape, name=None, init=False, end=False):
        with tf.variable_scope(name) as scope:
            if init: shape[0] = int(np.prod(input.get_shape()[1:]))

            fcw = tf.Variable(tf.truncated_normal(shape, dtype=tf.float32, stddev=1e-1), name='weights')
            fcb = tf.Variable(tf.constant(1.0, shape=[shape[1]], dtype=tf.float32),
                                 trainable=True, name='biases')

            if init: input = tf.reshape(input, [-1, shape[0]])
            fcl = tf.nn.bias_add(tf.matmul(input, fcw), fcb)
            if end:
                self.fc[name] = tf.nn.softmax(fcl)
            else:
                self.fc[name] = tf.nn.relu(fcl)
            self.parameters += [fcw, fcb]
        return self.fc[name]


    def encoder(self, imgs, name, reuse = False):
        self.parameters = []
        with tf.variable_scope(name, reuse = reuse) as scope:

            images = self.preprocess(imgs,name='preprocess')

            conv1_1 = self.conv2d(input = images,  filter_shape = [3, 3, 4,   64],  name = "conv1_1")
            conv1_2 = self.conv2d(input = conv1_1, filter_shape = [3, 3, 64,  64],  name = "conv1_2")
            pool1 = self.max_pool(input = conv1_2, name = "pool1")

            conv2_1 = self.conv2d(input = pool1,   filter_shape = [3, 3, 64,  128], name = "conv2_1")
            conv2_2 = self.conv2d(input = conv2_1, filter_shape = [3, 3, 128, 128], name = "conv2_2")
            pool2 = self.max_pool(input = conv2_2, name = "pool2")

            conv3_1 = self.conv2d(input = pool2,   filter_shape = [3, 3, 128, 256], name = "conv3_1")
            conv3_2 = self.conv2d(input = conv3_1, filter_shape = [3, 3, 256, 256], name = "conv3_2")
            conv3_3 = self.conv2d(input = conv3_2, filter_shape = [3, 3, 256, 256], name = "conv3_3")
            pool3 = self.max_pool(input = conv3_3, name = "pool3")

            conv4_1 = self.conv2d(input = pool3,   filter_shape = [3, 3, 256, 512], name = "conv4_1")
            conv4_2 = self.conv2d(input = conv4_1, filter_shape = [3, 3, 512, 512], name = "conv4_2")
            conv4_3 = self.conv2d(input = conv4_2, filter_shape = [3, 3, 512, 512], name = "conv4_3")
            pool4 = self.max_pool(input = conv4_3, name = "pool4")

            conv5_1 = self.conv2d(input = pool4,   filter_shape = [3, 3, 512, 512], name = "conv5_1")
            conv5_2 = self.conv2d(input = conv5_1, filter_shape = [3, 3, 512, 512], name = "conv5_2")
            conv5_3 = self.conv2d(input = conv5_2, filter_shape = [3, 3, 512, 512], name = "conv5_3")
            pool5 = self.max_pool(input = conv5_3, name = "pool5")

            '''
            fc1 = self.fc_layer(input = pool5, shape=[None, 4096], name = "fc1", init=True)
            fc2 = self.fc_layer(input = fc1,   shape=[4096, 4096], name = "fc2")
            fc3 = self.fc_layer(input = fc2,   shape=[4096, 1000], name = "fc3", end=True)
            '''

            return pool5

#    def decoder(self, )

if __name__ == '__main__':
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16(imgs, 'DeepMatting/vgg16_weights.npz', sess)

    img1 = misc.imread('DeepMatting/face.jpg', mode='RGB') # example of image
    img1 = misc.imresize(img1, (224, 224))

    prob = sess.run(vgg.probs, feed_dict={vgg.imgs: [img1]})[0]
    preds = (np.argsort(prob)[::-1])[0:5]
    for p in preds:
        print(class_names[p], prob[p])
