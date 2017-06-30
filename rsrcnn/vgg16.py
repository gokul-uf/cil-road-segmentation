# imports
import tensorflow as tf
from scipy import misc
import numpy as np
from imagenet_classes import class_names
import sys
import os
import time
from os import listdir
from scipy import ndimage
import random
from tqdm import tqdm
import time
import re

tf.app.flags.DEFINE_float("learning_rate"               , 1e-6 , "Learning rate.")
tf.app.flags.DEFINE_float("momentum"                    , 0.9  , "Momentum")
tf.app.flags.DEFINE_float("max_gradient_norm"           , 5.0   , "Clip gradients to this norm.")

tf.app.flags.DEFINE_integer("batch_size"                , 10    , "batch size.")
tf.app.flags.DEFINE_integer("num_epochs"                , 5000  , "number of epochs.")

#tf.app.flags.DEFINE_string("IMAGES_PATH"              , "./data/CIL/generate/patches/sat/", "path to images.")
tf.app.flags.DEFINE_string("IMAGES_PATH"              , "./data/CIL/overlap/sat/", "path to images.")
tf.app.flags.DEFINE_string("TEST_IMAGES_PATH"         , "./rsrcnn/kaggle/test_set_patches/", "path to patches from test images.")
tf.app.flags.DEFINE_string("TEST_OUTPUT_IMAGES_PATH"  , "./rsrcnn/kaggle/test_set_output/", "path to images generated from test images.")
#tf.app.flags.DEFINE_string("GROUNDTRUTHS_PATH"        , "./data/CIL/generate/patches/org/", "path to labels.")
tf.app.flags.DEFINE_string("GROUNDTRUTHS_PATH"        , "./data/CIL/overlap/org/", "path to labels.")
#tf.app.flags.DEFINE_string("DISTANCES_PATH"           , "./data/CIL/generate/patches/dst/", "path to distances.")
tf.app.flags.DEFINE_string("DISTANCES_PATH"           , "./data/CIL/overlap/dst/", "path to distances.")
tf.app.flags.DEFINE_string("WEIGHTS_PATH"             , "./rsrcnn/vgg16_c1-c13_weights", "path to weights.")
tf.app.flags.DEFINE_string("train_dir"                , "./rsrcnn/train_dir/", "Directory to save trained model.")
tf.app.flags.DEFINE_string("output_dir"               , "./rsrcnn/outputs/", "Directory to save output images.")
tf.app.flags.DEFINE_string("summaries_dir"            , "./data/CIL/generate/summaries", "path to summaries.")

tf.set_random_seed(1)

FLAGS = tf.app.flags.FLAGS


class rsrcnn:

	def __init__(self, weights=None, sess=None):

		if weights is None:

			self.sess = sess
			return

		self.sess = sess
		self.batch_size = FLAGS.batch_size
		self.inp_dim = 200
		self.output = None
		self.output_image = None

		self.is_training = tf.placeholder(tf.bool, name='is_training')
		self.keep_prob   = tf.placeholder(tf.float32, name='keep_prob')

		#Setup Learning Rate Decay
		self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
		self.starter_learning_rate = tf.constant(FLAGS.learning_rate, dtype=tf.float64)
		self.learning_rate = tf.train.exponential_decay(self.starter_learning_rate, self.global_step,
		648*20, 0.90, staircase=True)

		self.momentum = FLAGS.momentum
		self.max_gradient_norm = FLAGS.max_gradient_norm

		self.imgs         = tf.placeholder(tf.float32, [self.batch_size, self.inp_dim, self.inp_dim, 3])
		self.groundtruths = tf.placeholder(tf.float32, [self.batch_size, self.inp_dim, self.inp_dim])
		self.distances    = tf.placeholder(tf.float32, [self.batch_size, self.inp_dim, self.inp_dim])

		self.conv = {}
		self.pool = {}
		self.fc   = {}

		self.build_model('rsrcnn')

		if weights is not None and sess is not None:
			self.load_vgg16_weights(weights, 'rsrcnn')

		self.build_optimizer()

		self.saver = tf.train.Saver(max_to_keep=5)

	def load_vgg16_weights(self, weights_dir, name=None):

		conv1_1_b_np = np.load(os.path.join(weights_dir, "conv1_1_b.npy"))
		conv1_1_W_np = np.load(os.path.join(weights_dir, "conv1_1_W.npy"))
		conv1_2_b_np = np.load(os.path.join(weights_dir, "conv1_2_b.npy"))
		conv1_2_W_np = np.load(os.path.join(weights_dir, "conv1_2_W.npy"))

		conv2_1_b_np = np.load(os.path.join(weights_dir, "conv2_1_b.npy"))
		conv2_1_W_np = np.load(os.path.join(weights_dir, "conv2_1_W.npy"))
		conv2_2_b_np = np.load(os.path.join(weights_dir, "conv2_2_b.npy"))
		conv2_2_W_np = np.load(os.path.join(weights_dir, "conv2_2_W.npy"))

		conv3_1_b_np = np.load(os.path.join(weights_dir, "conv3_1_b.npy"))
		conv3_1_W_np = np.load(os.path.join(weights_dir, "conv3_1_W.npy"))
		conv3_2_b_np = np.load(os.path.join(weights_dir, "conv3_2_b.npy"))
		conv3_2_W_np = np.load(os.path.join(weights_dir, "conv3_2_W.npy"))
		conv3_3_b_np = np.load(os.path.join(weights_dir, "conv3_3_b.npy"))
		conv3_3_W_np = np.load(os.path.join(weights_dir, "conv3_3_W.npy"))

		conv4_1_b_np = np.load(os.path.join(weights_dir, "conv4_1_b.npy"))
		conv4_1_W_np = np.load(os.path.join(weights_dir, "conv4_1_W.npy"))
		conv4_2_b_np = np.load(os.path.join(weights_dir, "conv4_2_b.npy"))
		conv4_2_W_np = np.load(os.path.join(weights_dir, "conv4_2_W.npy"))
		conv4_3_b_np = np.load(os.path.join(weights_dir, "conv4_3_b.npy"))
		conv4_3_W_np = np.load(os.path.join(weights_dir, "conv4_3_W.npy"))

		conv5_1_b_np = np.load(os.path.join(weights_dir, "conv5_1_b.npy"))
		conv5_1_W_np = np.load(os.path.join(weights_dir, "conv5_1_W.npy"))
		conv5_2_b_np = np.load(os.path.join(weights_dir, "conv5_2_b.npy"))
		conv5_2_W_np = np.load(os.path.join(weights_dir, "conv5_2_W.npy"))
		conv5_3_b_np = np.load(os.path.join(weights_dir, "conv5_3_b.npy"))
		conv5_3_W_np = np.load(os.path.join(weights_dir, "conv5_3_W.npy"))


		with tf.variable_scope(name, reuse=True) as scope:

			conv1_1_b = tf.get_variable(initializer=tf.constant(conv1_1_b_np), name='conv1_1/biases')
			conv1_1_W = tf.get_variable(initializer=tf.constant(conv1_1_W_np), name='conv1_1/weights')
			conv1_2_b = tf.get_variable(initializer=tf.constant(conv1_2_b_np), name='conv1_2/biases')
			conv1_2_W = tf.get_variable(initializer=tf.constant(conv1_2_W_np), name='conv1_2/weights')

			conv2_1_b = tf.get_variable(initializer=tf.constant(conv2_1_b_np), name='conv2_1/biases')
			conv2_1_W = tf.get_variable(initializer=tf.constant(conv2_1_W_np), name='conv2_1/weights')
			conv2_2_b = tf.get_variable(initializer=tf.constant(conv2_2_b_np), name='conv2_2/biases')
			conv2_2_W = tf.get_variable(initializer=tf.constant(conv2_2_W_np), name='conv2_2/weights')

			conv3_1_b = tf.get_variable(initializer=tf.constant(conv3_1_b_np), name='conv3_1/biases')
			conv3_1_W = tf.get_variable(initializer=tf.constant(conv3_1_W_np), name='conv3_1/weights')
			conv3_2_b = tf.get_variable(initializer=tf.constant(conv3_2_b_np), name='conv3_2/biases')
			conv3_2_W = tf.get_variable(initializer=tf.constant(conv3_2_W_np), name='conv3_2/weights')
			conv3_2_b = tf.get_variable(initializer=tf.constant(conv3_2_b_np), name='conv3_2/biases')
			conv3_2_W = tf.get_variable(initializer=tf.constant(conv3_2_W_np), name='conv3_2/weights')

			conv4_1_b = tf.get_variable(initializer=tf.constant(conv4_1_b_np), name='conv4_1/biases')
			conv4_1_W = tf.get_variable(initializer=tf.constant(conv4_1_W_np), name='conv4_1/weights')
			conv4_2_b = tf.get_variable(initializer=tf.constant(conv4_2_b_np), name='conv4_2/biases')
			conv4_2_W = tf.get_variable(initializer=tf.constant(conv4_2_W_np), name='conv4_2/weights')
			conv4_2_b = tf.get_variable(initializer=tf.constant(conv4_2_b_np), name='conv4_2/biases')
			conv4_2_W = tf.get_variable(initializer=tf.constant(conv4_2_W_np), name='conv4_2/weights')

			conv5_1_b = tf.get_variable(initializer=tf.constant(conv5_1_b_np), name='conv5_1/biases')
			conv5_1_W = tf.get_variable(initializer=tf.constant(conv5_1_W_np), name='conv5_1/weights')
			conv5_2_b = tf.get_variable(initializer=tf.constant(conv5_2_b_np), name='conv5_2/biases')
			conv5_2_W = tf.get_variable(initializer=tf.constant(conv5_2_W_np), name='conv5_2/weights')
			conv5_2_b = tf.get_variable(initializer=tf.constant(conv5_2_b_np), name='conv5_2/biases')
			conv5_2_W = tf.get_variable(initializer=tf.constant(conv5_2_W_np), name='conv5_2/weights')

		print("params of C1-13 of vgg16 successfully loaded!")

	def conv2d(self, input, filter_shape, strides = (1,1,1,1), pad = "SAME", name = None, stddev=1e-1, drop_out = False):
		#print("In conv2d")
		with tf.variable_scope(name, reuse=True) as scope:
			kernel = tf.get_variable(initializer= tf.contrib.layers.xavier_initializer_conv2d(), name='weights')
			conv = tf.nn.conv2d(input, kernel, strides, padding=pad)

			if drop_out:
				conv = tf.nn.dropout(conv, keep_prob=self.keep_prob)

			biases = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(), name='biases')
			self.conv[name] = tf.nn.bias_add(conv, biases)

			self.parameters += [kernel, biases]
			return self.conv[name]

	# no padding in deconv layer
	# filter_shape => [batch, row, col]
	# input_shape  => [batch, row, col]
	def deconv2d(self, input, filter_shape, output_shape, strides = (1,2,2,1), pad = 'SAME', name = None, stddev=1e-1, drop_out = False):

		with tf.variable_scope(name, reuse=True) as scope:
			filter = tf.get_variable(initializer = tf.contrib.layers.xavier_initializer_conv2d(),
									name='weights')

			out = tf.nn.conv2d_transpose(value=input, filter=filter, output_shape=output_shape,
					strides=strides, padding=pad)

			if drop_out:
				out = tf.nn.dropout(out, keep_prob=self.keep_prob)

			return out

	def batch_norm(self, input, relu = True, name=None):

		with tf.variable_scope(name) as scope:
			bn_out = tf.layers.batch_normalization(input, training=self.is_training)

			if relu:
				return tf.nn.relu(bn_out, name="ReLU")
			else:
				return bn_out

	# no padding in deconv layer
	# filter_shape => [batch, row, col]
	# input_shape  => [batch, row, col]
	def deconv2d_custom(self, inp_tensor, filter_shape, stride = 3, name = None, stddev=1e-1):

		with tf.variable_scope(name, reuse=True) as scope:

			batch_size, input_rows, input_cols, _ = tf.unstack(tf.shape(inp_tensor))

			fil_rows = filter_shape[0]
			fil_cols = filter_shape[1]

			filter = tf.get_variable( initializer=tf.contrib.layers.xavier_initializer_conv2d(), name='weights')


			#filter = tf.constant([[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4]], dtype="float32")

			output_rows = (input_rows-1)*stride + fil_rows

			output = tf.ones([batch_size, output_rows, output_rows], dtype="float32")

			row_num = 0

			_, num_rows, num_cols, _ = inp_tensor.get_shape();

			# same number of rows and columns in input
			for i in range(0, num_rows):

				col_num = 0

				for j in range(0, num_cols):

					r_fil = tf.reshape( filter, [1, -1] )
					r_inp = tf.reshape( inp_tensor[:, i, j], [-1, 1] )

					cur_inps_conv = tf.matmul(r_inp, r_fil)
					cur_inps_conv = tf.reshape( cur_inps_conv, [-1, fil_rows, fil_cols] )
					#print(cur_inps_conv.eval())

					padded_out = self.pad_zeroes(cur_inps_conv, row_num*stride, col_num*stride, fil_rows, output_rows)
					col_num += 1

					output = output + padded_out
					# print("Output")
					# print(output.eval())
					# input()

				row_num += 1

		return output

	# paddings => [[inside, outside], [top, bottom], [left, right]]
	def pad_zeroes(self, tensor, row_pos, col_pos, tensor_dim, output_dim):

		# print("dimensions")
		# print(output_dim)
		# print(row_pos)
		# print(col_pos)
		# print(tensor_dim)

		bottom_pad = tf.cast(output_dim-(tensor_dim+row_pos), dtype=tf.int32 )
		right_pad  = tf.cast(output_dim-(tensor_dim+col_pos), dtype=tf.int32 )

		# print("padding shape")
		# print(dim_row.eval())
		# print(dim_col.eval())

		paddings = tf.Variable([ [0, 0], [row_pos, bottom_pad], [col_pos, right_pad] ], dtype=tf.int32, name="paddings")

		#self.sess.run(tf.global_variables_initializer())

		padded_out = tf.pad(tensor, paddings, "CONSTANT")

		return padded_out

	def upsample_2d(self, input, size, name):
	  with tf.variable_scope(name) as scope:
		  return tf.image.resize_nearest_neighbor(input, size = size, name = name)

	def max_pool(self, input, ksize=[1,2,2,1], strides = [1,2,2,1], padding = "SAME", name = None):
		#print("In max_pool")
		with tf.variable_scope(name) as scope:
			res = tf.nn.max_pool(value=input, ksize=ksize, strides=strides, padding=padding)
		self.pool[name] = res
		return res

	def fusion(self, deconv_input, conv_input, name):
		with tf.variable_scope(name) as scope:

			#if tf.not_equal(tf.shape(deconv_input), tf.shape(conv_input)):
			if deconv_input.shape.as_list() != conv_input.shape.as_list():

				# cut down shape of conv_output to that of deconv_output
				if deconv_input.shape.as_list()[1] < conv_input.shape.as_list()[1]:

					fraction = deconv_input.shape.as_list()[1] / conv_input.shape.as_list()[1]

					crop_list = []
					for b in range(self.batch_size):
						crop_output = tf.image.central_crop(conv_input[b], central_fraction = fraction)
						crop_list.append(crop_output)

					crop_output = tf.stack(crop_list)

					return tf.add(crop_output, deconv_input)

				else:
					fraction = conv_input.shape.as_list()[1] / deconv_input.shape.as_list()[1]
					
					crop_list = []
					for b in range(self.batch_size):
						crop_output = tf.image.central_crop(deconv_input[b], central_fraction = fraction)
						crop_list.append(crop_output)

					crop_output = tf.stack(crop_list)

					return tf.add(crop_output, conv_input)


			else:

				return tf.add(conv_input, deconv_input)

	def crop(self, input_tensor, name=None):

		with tf.variable_scope(name) as scope:

			fraction = self.inp_dim / input_tensor.shape.as_list()[1]

			crop_list = []

			for b in range(self.batch_size):
				crop_output = tf.image.central_crop(input_tensor[b], central_fraction = fraction)
				crop_list.append(crop_output)

			crop_output = tf.stack(crop_list)

			return crop_output

	def preprocess(self, name=None):
		with tf.variable_scope(name) as scope:
			mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
		return self.imgs - mean

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


	# Pass the groundtruth tf, image_index for distances
	# the loss is unnormalized
	def overall_loss(self):

		# exp_dists = tf.exp(-self.distances)

		# groundtruths_cmpl = (1-self.groundtruths)

		# loss = ( exp_dists * self.output * groundtruths_cmpl ) + \
		# 	( tf.log(1+tf.exp(-self.output)) * (self.groundtruths + (groundtruths_cmpl*exp_dists) ) )

		#loss = tf.maximum(self.output, 0) - (self.output * self.groundtruths) + tf.log(1 + tf.exp(-tf.abs(self.output)))

		#loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = self.groundtruths, logits = self.output)

		exp_dists = tf.exp(self.distances)
		loss = tf.nn.weighted_cross_entropy_with_logits(
													targets = self.groundtruths,
													logits  = self.output,
													pos_weight = exp_dists,
													name=None
												)

		# loss_2 = tf.losses.hinge_loss(
		# 							labels = self.groundtruths,
		# 							logits = self.output,
		# 							reduction = NONE
		# 							)

		# loss = tf.add(loss_1, loss_2, name="total_loss")


		return tf.reduce_mean( tf.reduce_sum(loss, axis=[1,2]) )

	def build_model(self, name, reuse = False):

		self.parameters = []
		with tf.variable_scope(name, reuse=reuse) as scope:

			self.initialize_all_variables()

			images = self.preprocess(name='preprocess')

			conv1_1 = self.conv2d    (input = images,  filter_shape = [3, 3, 3,   64],  name = "conv1_1")
			bn1_1   = self.batch_norm(input = conv1_1, name = "bn1_1")
			conv1_2 = self.conv2d    (input =   bn1_1, filter_shape = [3, 3, 64,  64],  name = "conv1_2")
			bn1_2   = self.batch_norm(input = conv1_2, name = "bn1_2")
			pool1   = self.max_pool  (input =   bn1_2, name = "pool1")


			conv2_1 = self.conv2d    (input = pool1,   filter_shape = [3, 3, 64,  128], name = "conv2_1")
			bn2_1   = self.batch_norm(input = conv2_1, name = "bn2_1")
			conv2_2 = self.conv2d    (input =   bn2_1, filter_shape = [3, 3, 128, 128], name = "conv2_2")
			bn2_2   = self.batch_norm(input = conv2_2, name = "bn2_2")
			pool2   = self.max_pool  (input =   bn2_2, name = "pool2")


			conv3_1 = self.conv2d    (input = pool2,   filter_shape = [3, 3, 128, 256], name = "conv3_1")
			bn3_1   = self.batch_norm(input = conv3_1, name = "bn3_1")
			conv3_2 = self.conv2d    (input =   bn3_1, filter_shape = [3, 3, 256, 256], name = "conv3_2")
			bn3_2   = self.batch_norm(input = conv3_2, name = "bn3_2")
			conv3_3 = self.conv2d    (input =   bn3_2, filter_shape = [3, 3, 256, 256], name = "conv3_3")
			bn3_3   = self.batch_norm(input = conv3_3, name = "bn3_3")
			pool3   = self.max_pool  (input =   bn3_3, name = "pool3")

			conv4_1 = self.conv2d    (input = pool3,   filter_shape = [3, 3, 256, 512], name = "conv4_1")
			bn4_1   = self.batch_norm(input = conv4_1, name = "bn4_1")
			conv4_2 = self.conv2d    (input =   bn4_1, filter_shape = [3, 3, 512, 512], name = "conv4_2")
			bn4_2   = self.batch_norm(input = conv4_2, name = "bn4_2")
			conv4_3 = self.conv2d    (input =   bn4_2, filter_shape = [3, 3, 512, 512], name = "conv4_3")
			bn4_3   = self.batch_norm(input = conv4_3, name = "bn4_3")
			pool4   = self.max_pool  (input =   bn4_3, name = "pool4")


			conv5_1 = self.conv2d    (input = pool4,   filter_shape = [3, 3, 512, 512], name = "conv5_1")
			bn5_1   = self.batch_norm(input = conv5_1, name = "bn5_1")
			conv5_2 = self.conv2d    (input =   bn5_1, filter_shape = [3, 3, 512, 512], name = "conv5_2")
			bn5_2   = self.batch_norm(input = conv5_2, name = "bn5_2")
			conv5_3 = self.conv2d    (input =   bn5_2, filter_shape = [3, 3, 512, 512], name = "conv5_3")
			bn5_3   = self.batch_norm(input = conv5_3, name = "bn5_3")
			pool5   = self.max_pool  (input =   bn5_3, name = "pool5")


			# No padding in c14-18 and DCs
			conv14 = self.conv2d    (input = pool5,  filter_shape = [7, 7, 512,  2048], name = "conv14")
			bn14   = self.batch_norm(input = conv14, name = "bn14")
			conv15 = self.conv2d    (input =   bn14, filter_shape = [1, 1, 2048, 512],  name = "conv15")
			bn15   = self.batch_norm(input = conv15, name = "bn15")
			conv16 = self.conv2d    (input =   bn15,  filter_shape = [1, 1, 512,  1],    name = "conv16")
			bn16   = self.batch_norm(input = conv16, name = "bn16")


			conv17 = self.conv2d    (input = pool4,   filter_shape = [1, 1, 512,  1],    name = "conv17")
			bn17   = self.batch_norm(input = conv17, name = "bn17")


			conv18 = self.conv2d    (input = pool3, filter_shape = [1, 1, 256,  1],    name = "conv18")
			bn18   = self.batch_norm(input = conv18, name = "bn18")



			deconv1 = self.deconv2d  (input = bn16   , filter_shape=[4, 4, 1, 1], output_shape=[self.batch_size, 13, 13, 1], name="deconv_1")
			bn_dc_1 = self.batch_norm(input = deconv1, name = "bn_dc_1")

			fusion1 = self.fusion(bn_dc_1, bn17, name="fusion_1")

			deconv2 = self.deconv2d  (input = fusion1, filter_shape=[4, 4, 1, 1], output_shape=[self.batch_size, 25, 25, 1], name="deconv_2")
			bn_dc_2 = self.batch_norm(input = deconv2, name = "bn_dc_2")

			fusion2 = self.fusion(bn_dc_2, bn18, name="fusion_2")

			deconv3 = self.deconv2d  (input = fusion2, filter_shape=[16, 16, 1, 1], output_shape=[self.batch_size, 220, 220, 1], strides=(1,9,9,1), name="deconv_3")
			bn_dc_3 = self.batch_norm(input = deconv3, name = "bn_dc_3", relu = False)


			crop   = self.crop (bn_dc_3, name="crop")
			output = tf.reshape(crop , shape=[self.batch_size, self.inp_dim, self.inp_dim] )


			self.output_image = crop
			self.output = output

			print("building model done")

	def build_optimizer(self):

		self.loss = self.overall_loss()
		tf.summary.histogram("output_weights", self.output)
		tf.summary.image("output_map", self.output_image)
		tf.summary.scalar('loss', self.loss)
		print("loss shape")
		print(self.loss.get_shape())

		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):

			#self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
			self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=self.momentum)
			#self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
			#self.optimizer = tf.train.RMSPropOptimizer(learning_rate=FLAGS.learning_rate)
			

			# self.gradients = self.optimizer.compute_gradients(self.loss)

			# self.capped_gradients = [( tf.clip_by_value( grad, -self.max_gradient_norm, self.max_gradient_norm ), variable ) for
			# 																grad, variable in self.gradients if grad is not None]

			# self.train_op = self.optimizer.apply_gradients(self.capped_gradients)

			self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

	def initialize_variable(self, scope_name, var_name, shape):
		with tf.variable_scope(scope_name) as scope:
			v = tf.get_variable(var_name, shape)
			#print(v.name)
			#scope.reuse_variable()

	def initialize_all_variables(self):

		self.initialize_variable("conv1_1", "weights", [3, 3, 3,   64])
		self.initialize_variable("conv1_1", "biases" ,            [64])
		self.initialize_variable("conv1_2", "weights", [3, 3, 64,  64])
		self.initialize_variable("conv1_2", "biases" ,            [64])

		self.initialize_variable("conv2_1", "weights", [3, 3, 64,  128])
		self.initialize_variable("conv2_1", "biases" ,            [128])
		self.initialize_variable("conv2_2", "weights", [3, 3, 128, 128])
		self.initialize_variable("conv2_2", "biases" ,            [128])

		self.initialize_variable("conv3_1", "weights", [3, 3, 128, 256])
		self.initialize_variable("conv3_1", "biases" ,            [256])
		self.initialize_variable("conv3_2", "weights", [3, 3, 256, 256])
		self.initialize_variable("conv3_2", "biases" ,            [256])
		self.initialize_variable("conv3_3", "weights", [3, 3, 256, 256])
		self.initialize_variable("conv3_3", "biases" ,            [256])

		self.initialize_variable("conv4_1", "weights", [3, 3, 256, 512])
		self.initialize_variable("conv4_1", "biases" ,            [512])
		self.initialize_variable("conv4_2", "weights", [3, 3, 512, 512])
		self.initialize_variable("conv4_2", "biases" ,            [512])
		self.initialize_variable("conv4_3", "weights", [3, 3, 512, 512])
		self.initialize_variable("conv4_3", "biases" ,            [512])

		self.initialize_variable("conv5_1", "weights", [3, 3, 512, 512])
		self.initialize_variable("conv5_1", "biases" ,            [512])
		self.initialize_variable("conv5_2", "weights", [3, 3, 512, 512])
		self.initialize_variable("conv5_2", "biases" ,            [512])
		self.initialize_variable("conv5_3", "weights", [3, 3, 512, 512])
		self.initialize_variable("conv5_3", "biases" ,            [512])

		self.initialize_variable("conv14", "weights", [7, 7, 512,  2048])
		self.initialize_variable("conv14", "biases" ,             [2048])
		self.initialize_variable("conv15", "weights", [1, 1, 2048, 512])
		self.initialize_variable("conv15", "biases" ,             [512])

		self.initialize_variable("conv16", "weights", [1, 1, 512,  1])
		self.initialize_variable("conv16", "biases" ,             [1])
		self.initialize_variable("conv17", "weights", [1, 1, 512,  1])
		self.initialize_variable("conv17", "biases" ,             [1])
		self.initialize_variable("conv18", "weights", [1, 1, 256,  1])
		self.initialize_variable("conv18", "biases" ,             [1])


		self.initialize_variable("deconv_1", "weights", [4, 4, 1, 1])
		self.initialize_variable("deconv_2", "weights", [4, 4, 1, 1])
		self.initialize_variable("deconv_3", "weights", [16, 16, 1, 1])

	def save(self, sess, epoch):
		file_name = "epoch_" + str(epoch) + ".ckpt"
		self.checkpoint_path = os.path.join(FLAGS.train_dir, file_name)
		self.saver.save(sess, self.checkpoint_path)

def test_deconv2d_custom():

	sess = tf.InteractiveSession()
	model = rsrcnn(sess=sess)

	inp_tensor = tf.constant([
								[[1,1,1],[1,1,1],[1,1,1]],
								[[2,2,2],[2,2,2],[2,2,2]]
							],
							 dtype="float32")

	model.initialize_variable("deconv_1", "weights", [4,4])
	model.initialize_variable("deconv_2", "weights", [4,4])

	model.deconv2d_custom(inp_tensor, filter_shape=[4,4], name="deconv_1")
	model.deconv2d_custom(inp_tensor, filter_shape=[4,4], name="deconv_2")

def f_function(distance):

	distance = np.sqrt(distance)
	max_dist = np.amax(distance)
	threshold = 0.3 * max_dist

	distance[distance == 0] = 0
	distance[(distance > 0) & (distance < threshold)] /= max_dist
	distance[distance > threshold] = threshold / max_dist

	return distance

def read_data():

	print("Reading distances")
	no_road_images = []
	distances = []
	# for now, no distances used. using ground truths as dummy distances to run the code
	for file in listdir(FLAGS.DISTANCES_PATH):
		distance_image = ndimage.imread(FLAGS.DISTANCES_PATH + file, mode = 'L')
		distances.append(f_function(distance_image))
	print("Number of distances: {0}".format(len(distances)))

	print("Reading images")
	images = []
	for file in listdir(FLAGS.IMAGES_PATH):
		image = ndimage.imread(FLAGS.IMAGES_PATH + file)
		images.append(image)
	print("Number of images: {0}".format(len(images)))

	print("Reading labels")
	groundtruths = []
	for file in listdir(FLAGS.GROUNDTRUTHS_PATH):
		groundtruth = ndimage.imread(FLAGS.GROUNDTRUTHS_PATH + file, mode = 'L')
		groundtruth[groundtruth < 127] = 0
		groundtruth[groundtruth >= 127] = 1
		groundtruths.append(groundtruth)
	print("Number of groundtruths: {0}".format(len(groundtruths)))

	


	print("Randomizing inputs")
	# randomly permute
	zipped_list = list(zip(images, groundtruths, distances))
	np.random.shuffle(zipped_list)

	images, groundtruths, distances = zip(*zipped_list)

	return (images, groundtruths, distances)

def read_test_data():

	print("Reading test images")
	
	# sort files by file names
	sorted_file_names = []
	for file in listdir(FLAGS.TEST_IMAGES_PATH):
		sorted_file_names.append(file)

	sort_nicely(sorted_file_names)

	images = []
	for file in sorted_file_names:	
		
		image = ndimage.imread(FLAGS.TEST_IMAGES_PATH + file)
		images.append(image)
	print("Number of images: {0}".format(len(images)))

	return images

def train(sess, model, train_images, train_groundtruths, train_distances, val_images, val_groundtruths, val_distances, load=None):

	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
									 sess.graph)
	test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')

	if load is None:
		model.sess.run(tf.global_variables_initializer())

	else:

		# load trained model to further train
		tf.reset_default_graph()

		try:
			model_path = os.path.join(FLAGS.train_dir, "epoch_12.ckpt")
			print("Reading model parameters from {0}".format(model_path))
			model.saver.restore(sess, model_path)

		except:
			print("Trained model not found. Exiting!")
			sys.exit()


	print("All variables initialized.")

	print("Learning rate={0}".format(FLAGS.learning_rate))

	print("Starting training")

	val_loss_last_2_epochs = [float("inf"), float("inf")]

	print("train set length={0}".format(len(train_images)))

	for epoch in range(FLAGS.num_epochs):

		print("Epoch {0} started".format(epoch))
		sys.stdout.flush()

		start = time.time()

		train_loss = 0
		lr = 0
		# iterate on batches
		for i in tqdm(range(len(train_images) // FLAGS.batch_size)):

			fd = {	model.distances    : train_distances[i * FLAGS.batch_size: (i + 1) * FLAGS.batch_size],
					model.groundtruths : train_groundtruths[i * FLAGS.batch_size: (i + 1) * FLAGS.batch_size],
					model.imgs         : train_images[i * FLAGS.batch_size: (i + 1) * FLAGS.batch_size],
					model.is_training  : 1,
					model.keep_prob    : 0.8
				}

			_, train_loss, summary, lr = sess.run([model.train_op, model.loss, merged, model.learning_rate], feed_dict=fd)

			train_writer.add_summary(summary, epoch*648 + i)

		end = time.time()
		print("Epoch {0} done. Time take = {1}".format( epoch, (end-start)/60 ))
		print("Learning rate = {0}".format(lr))
		print("training loss = {0}".format(train_loss))
		sys.stdout.flush()

		val_losses = []
		# validate on validation set
		for i in tqdm(range(len(val_images) // FLAGS.batch_size)):

			fd = {	model.distances    : val_distances[i * FLAGS.batch_size: (i + 1) * FLAGS.batch_size],
					model.groundtruths : val_groundtruths[i * FLAGS.batch_size: (i + 1) * FLAGS.batch_size],
					model.imgs         : val_images[i * FLAGS.batch_size: (i + 1) * FLAGS.batch_size],
					model.is_training  : 0,
					model.keep_prob    : 1.0
					}

			output, loss, summary = sess.run([model.output, model.loss, merged], feed_dict=fd)
			val_losses.append(loss)
			test_writer.add_summary(summary, epoch*648 + i)

		avg_val_loss = sum(val_losses)/len(val_losses)
		print( "validation loss = {0}".format(avg_val_loss) )
		sys.stdout.flush()

		if epoch%3 == 0:
			model.save(sess, epoch)

		# if epoch!=0 and epoch%5 == 0:
		# 	FLAGS.learning_rate /= 2.0

		# exit if validation loss starts increasing
		# if avg_val_loss > val_loss_last_2_epochs[1]  and avg_val_loss > val_loss_last_2_epochs[0]:

		# 	print("Epoch {0}: avg val loss greater than last 2 epoch. Saving model and exiting!".format(epoch))
		# 	sys.exit()

		# else:
		#	val_loss_last_2_epochs[0], val_loss_last_2_epochs[1] = val_loss_last_2_epochs[1], avg_val_loss


		# Shuffle the training set again
		zipped_list = list(zip(train_images, train_groundtruths, train_distances))
		np.random.shuffle(zipped_list)
		train_images, train_groundtruths, train_distances = zip(*zipped_list)

def test(sess, model, val_images, val_groundtruths, val_distances):

	tf.reset_default_graph()

	try:
		model_path = os.path.join(FLAGS.train_dir, "epoch_180/epoch_180.ckpt")
		print("Reading model parameters from {0}".format(model_path))
		model.saver.restore(sess, model_path)

	except:
		print("Trained model not found. Exiting!")
		sys.exit()

	outputs_list = []
	for i in tqdm(range(len(val_images) // FLAGS.batch_size)):

		fd = {	
				model.imgs         : val_images[i * FLAGS.batch_size: (i + 1) * FLAGS.batch_size],
				model.is_training  : 0,
				model.keep_prob    : 1.0
				}

		output = sess.run(model.output, feed_dict=fd)
		outputs_list.append(output)

	# 1 implies road, so their pixel values changed to 255 while saving back as image
	for i in range(len(val_groundtruths)):
		val_groundtruths[i][ val_groundtruths[i] == 1 ] = 255

	image_num = 0
	for i in range(len(outputs_list)):

		outputs_list[i][ outputs_list[i] >= 0 ] = 255
		outputs_list[i][ outputs_list[i] <  0 ] = 0

		for j in range(FLAGS.batch_size):

			output = outputs_list[i][j]

			misc.imsave(os.path.join(FLAGS.output_dir + 'actual/', str(image_num) + '.jpg'), output)
			misc.imsave(os.path.join(FLAGS.output_dir + 'expected/', str(image_num) + '.jpg'), val_groundtruths[i*FLAGS.batch_size + j])
			image_num += 1

def test_submission(sess, model, test_images):

	tf.reset_default_graph()

	try:
		model_path = os.path.join(FLAGS.train_dir, "epoch_6.ckpt")
		print("Reading model parameters from {0}".format(model_path))
		model.saver.restore(sess, model_path)

	except:
		print("Trained model not found. Exiting!")
		sys.exit()

	outputs_list = []
	for i in tqdm(range(len(test_images) // FLAGS.batch_size)):

		fd = {	
				model.imgs 		  : test_images[i * FLAGS.batch_size: (i + 1) * FLAGS.batch_size],
				model.is_training : 0,
				model.keep_prob   : 1.0
			}

		output = sess.run(model.output, feed_dict=fd)

		# unpack patches in the current batch
		for img in range(FLAGS.batch_size):
			outputs_list.append(output[img])


	# convert to binary array
	image_num = 0
	for i in range(len(outputs_list)):

		outputs_list[i][ outputs_list[i] >= 0 ] = 1
		outputs_list[i][ outputs_list[i] <  0 ] = 0


	# create full 608x608 binary output images
	full_images = []
	image_num = 1
	for i in range(0,len(outputs_list),16):
		
		full_image = None

		for j in range(i,i+16,4):

			left  = np.hstack((outputs_list[j], outputs_list[j+1]))
			right = np.hstack((outputs_list[j+2], outputs_list[j+3][:, 192:])) # only last eight columns
			row   = np.hstack((left, right))

			if full_image is None:
				full_image = row

			else:
				if (j+4)%16==0:
					full_image = np.vstack(( full_image, row[192:, :] ))
				else:
					full_image = np.vstack((full_image, row))

		#full_images.append(full_image)
		misc.imsave(os.path.join(FLAGS.TEST_OUTPUT_IMAGES_PATH, str(image_num) + '.jpg'), full_image)
		image_num += 1

def tryint(s):
	try:
		return int(s)
	except:
		return s

def alphanum_key(s):
	""" Turn a string into a list of string and number chunks.
		"z23a" -> ["z", 23, "a"]
	"""
	return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
	""" Sort the given list in the way that humans expect.
	"""
	l.sort(key=alphanum_key)


if __name__ == '__main__':

	# test_deconv2d_custom()

	print("params passed:")

	if len(sys.argv) > 0:
		print(sys.argv)
	else:
		print("None")

	

	# running test on dev set
	if 'test' in sys.argv:
		sess = tf.Session()

		print("Creating model")
		model = rsrcnn(FLAGS.WEIGHTS_PATH, sess)
		#tf.summary.image('image-output', tf.expand_dims(model.output, -1))

		images, groundtruths, distances = read_data()

		# number of total patches = 381
		# validation = 21
		# training = 360

		val_images = images[0:21]
		train_images = images[21:]

		val_groundtruths = groundtruths[0:21]
		train_groundtruths = groundtruths[21:]

		val_distances = distances[0:21]
		train_distances = distances[21:]

		test(sess, model, val_images, val_groundtruths, val_distances)


	elif 'submit' in sys.argv:
		sess = tf.Session()

		print("Creating model")
		model = rsrcnn(FLAGS.WEIGHTS_PATH, sess)
		#tf.summary.image('image-output', tf.expand_dims(model.output, -1))

		images = read_test_data()

		# number of total test patches = 800
		# image 1, patch 0,0     => 0
		# image 1, patch 0,200   => 1
		# image 1, patch 0,400   => 2
		# image 1, patch 0,408   => 3.......
		# image 1, patch 408,408 => 15
		
		test_submission(sess, model, images)

	else:

		load = None
		if 'load' in sys.argv:
			load = 'load'

		sess = tf.Session()

		print("Creating model")
		model = rsrcnn(FLAGS.WEIGHTS_PATH, sess)
		#tf.summary.image('image-output', tf.expand_dims(model.output, -1))

		images, groundtruths, distances = read_data()

		# number of total patches = 381
		# validation = 21
		# training = 360

		val_images = images[0:len(images)//10]
		train_images = images[len(images)//10:]

		val_groundtruths = groundtruths[0:len(images)//10]
		train_groundtruths = groundtruths[len(images)//10:]

		val_distances = distances[0:len(images)//10]
		train_distances = distances[len(images)//10:]

		train(sess, model, train_images, train_groundtruths, train_distances, val_images, val_groundtruths, val_distances, load)



	

