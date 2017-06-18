# imports
import tensorflow as tf
from scipy import misc
import numpy as np
from imagenet_classes import class_names

class rsrcnn:
    def __init__(self, imgs, weights=None, sess=None):
        self.imgs = imgs
        self.conv = {}
        self.pool = {}
        self.fc   = {}
        self.output = self.build_model(self.imgs, 'rsrcnn')
        if weights is not None and sess is not None:
            self.load_weights(weights, sess)

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        
        keys = sorted(weights)

        for i, k in enumerate(keys):  
            print(i, k, np.shape(weights[k]))
            sess.run(self.parameters[i].assign(weights[k]))
            if k == 'conv5_3_W': break
      

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
          
    def deconv2d(self, input, filter_shape, output_shape, strides = (1,2,2,1), pad = 'SAME', name = None):
        with tf.variable_scope(name) as scope:
            return tf.nn.conv2d_transpose(input, filter, output_shape, strides, pad, name)
    
    def upsample_2d(self, input, size, name):
      with tf.variable_scope(name) as scope:
          return tf.image.resize_nearest_neighbor(input, size = size, name = name)
        
    def max_pool(self, input, ksize=[1,2,2,1], strides = [1,2,2,1], padding = "SAME", name = None):
        with tf.variable_scope(name) as scope:
            res = tf.nn.max_pool(value=input, ksize=ksize, strides=strides, padding=padding)
        self.pool[name] = res
        return res
    
    def fusion(self, deconv_input, conv_input, filter_shape, name):
        with tf.variable_scope(name) as scope:
            if deconv_input.shape.as_list() != conv_input.shape.as_list():
                # cut down shape of conv_output to that of deconv_output
                if deconv_input.shape.as_list()[1] < conv_input.shape.as_list()[1]:
                    fraction = deconv_input.shape.as_list()[1] / conv_input.shape.as_list()[1]
                else:
                    fraction = conv_input.shape.as_list()[1] / deconv_input.shape.as_list()[1]
                crop_output = tf.image.central_crop(conv_input, fraction = fraction)
            else:
                crop_output = conv_input
            assert crop_output.shape.as_list() == deconv_input.shape.as_list()
            return tf.add(crop_output, deconv_input)

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


    def build_model(self, imgs, name, reuse = False):
        self.parameters = []
        with tf.variable_scope(name, reuse = reuse) as scope:

            images = self.preprocess(imgs,name='preprocess')

            conv1_1 = self.conv2d(input = images,  filter_shape = [3, 3, 3,   64],  name = "conv1_1")
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
            
            conv14 = self.conv2d(input = pool5,   filter_shape = [7, 7, 512,  2048], name = "conv14")
            conv15 = self.conv2d(input = conv14,  filter_shape = [7, 7, 2048, 512],  name = "conv15")
            conv16 = self.conv2d(input = conv14,  filter_shape = [1, 1, 512,  1],    name = "conv16")
            
            conv17 = self.conv2d(input = pool4,   filter_shape = [1, 1, 512,  1],    name = "conv17")
            conv18 = self.conv2d(input = pool3,   filter_shape = [1, 1, 256,  1],    name = "conv18")
            
            
            # deconv1 = self.upsample_2d(input = conv16, size = (?,?), name = '        ')
            # return pool5
            return True

#    def decoder(self, )

if __name__ == '__main__':
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    model = rsrcnn(imgs, 'vgg16_weights.npz', sess)

    img1 = misc.imread('hotdog.jpg', mode='RGB') # example of image
    img1 = misc.imresize(img1, (224, 224))

    prob = sess.run(model.oputput, feed_dict={model.imgs: [img1]})[0]
    preds = (np.argsort(prob)[::-1])[0:5]
    for p in preds:
        print(class_names[p], prob[p])

'''
(0, 'conv1_1_W', (3, 3, 3, 64))
(1, 'conv1_1_b', (64,))
(2, 'conv1_2_W', (3, 3, 64, 64))
(3, 'conv1_2_b', (64,))
(4, 'conv2_1_W', (3, 3, 64, 128))
(5, 'conv2_1_b', (128,))
(6, 'conv2_2_W', (3, 3, 128, 128))
(7, 'conv2_2_b', (128,))
(8, 'conv3_1_W', (3, 3, 128, 256))
(9, 'conv3_1_b', (256,))
(10, 'conv3_2_W', (3, 3, 256, 256))
(11, 'conv3_2_b', (256,))
(12, 'conv3_3_W', (3, 3, 256, 256))
(13, 'conv3_3_b', (256,))
(14, 'conv4_1_W', (3, 3, 256, 512))
(15, 'conv4_1_b', (512,))
(16, 'conv4_2_W', (3, 3, 512, 512))
(17, 'conv4_2_b', (512,))
(18, 'conv4_3_W', (3, 3, 512, 512))
(19, 'conv4_3_b', (512,))
(20, 'conv5_1_W', (3, 3, 512, 512))
(21, 'conv5_1_b', (512,))
(22, 'conv5_2_W', (3, 3, 512, 512))
(23, 'conv5_2_b', (512,))
(24, 'conv5_3_W', (3, 3, 512, 512))
(25, 'conv5_3_b', (512,))
(26, 'fc6_W', (25088, 4096))
(27, 'fc6_b', (4096,))
(28, 'fc7_W', (4096, 4096))
(29, 'fc7_b', (4096,))
(30, 'fc8_W', (4096, 1000))
(31, 'fc8_b', (1000,))
'''