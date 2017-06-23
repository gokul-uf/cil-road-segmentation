# imports
import tensorflow as tf
from scipy import misc
import numpy as np
from imagenet_classes import class_names
import sys
import os
from os import listdir
from scipy import ndimage
import random

tf.app.flags.DEFINE_float("learning_rate"             , 0.001 , "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm"         , 5.0   , "Clip gradients to this norm.")

FLAGS = tf.app.flags.FLAGS


IMAGES_PATH = "../data/mhui/generate/patches_rotation/images/"
GROUNDTRUTHS_PATH = "../data/mhui/generate/patches_rotation/groundtruth/"
DISTANCES_PATH = "../data/mhui/generate/patches_rotation/distance/"

IMAGE_HEIGHT = 200
IMAGE_WIDTH = 200
BATCH_SIZE = 10


class rsrcnn:

	def __init__(self, weights=None, sess=None):

		if weights is None:

			self.sess = sess
			return

		self.sess = sess
		self.batch_size = 10
		self.inp_dim = 200

		self.learning_rate = FLAGS.learning_rate
		self.max_gradient_norm = FLAGS.max_gradient_norm

		self.imgs         = tf.placeholder(tf.float32, [self.batch_size, self.inp_dim, self.inp_dim, 3])
		self.groundtruths = tf.placeholder(tf.float32, [self.batch_size, self.inp_dim, self.inp_dim])
		self.distances    = tf.placeholder(tf.float32, [self.batch_size, self.inp_dim, self.inp_dim])
		self.distances_max = tf.placeholder(tf.float32, [None])

		self.conv = {}
		self.pool = {}
		self.fc   = {}
		
		self.build_model('rsrcnn')

		if weights is not None and sess is not None:
			self.load_vgg16_weights(weights, 'rsrcnn')

		self.build_optimizer()

		self.compute_distaces()	

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

	def compute_distaces(self, name=None):
		self.distances_max = tf.reduce_max(self.distances, axis=[1,2])


	def conv2d(self, input, filter_shape, strides = (1,1,1,1), activation = tf.nn.relu, pad = "SAME", name = None, stddev=1e-1):
		#print("In conv2d")
		with tf.variable_scope(name, reuse=True) as scope:
			kernel = tf.get_variable(initializer=tf.truncated_normal(filter_shape, dtype=tf.float32,
													 stddev=stddev), name='weights')
			conv = tf.nn.conv2d(input, kernel, strides, padding=pad)

			biases = tf.get_variable( initializer=tf.constant(0.0, shape=[filter_shape[3]], dtype=tf.float32),
								 trainable=True, name='biases')
			out = tf.nn.bias_add(conv, biases)

			self.conv[name] = tf.nn.relu(out, name=scope.name)
			self.parameters += [kernel, biases]
			return self.conv[name]
		 
	# no padding in deconv layer
	# filter_shape => [batch, row, col]
	# input_shape  => [batch, row, col] 
	def deconv2d(self, input, filter_shape, output_shape, strides = (1,2,2,1), pad = 'SAME', name = None, stddev=1e-1):

		with tf.variable_scope(name, reuse=True) as scope:

			filter = tf.get_variable(initializer=tf.truncated_normal(filter_shape, 
									dtype=tf.float32, 
									stddev=stddev),
									name='weights')

			return tf.nn.conv2d_transpose(value=input, filter=filter, output_shape=output_shape, 
				strides=strides, padding=pad)

	# no padding in deconv layer
	# filter_shape => [batch, row, col]
	# input_shape  => [batch, row, col]
	def deconv2d_custom(self, inp_tensor, filter_shape, stride = 3, name = None, stddev=1e-1):

		with tf.variable_scope(name, reuse=True) as scope:

			batch_size, input_rows, input_cols, _ = tf.unstack(tf.shape(inp_tensor))

			fil_rows = filter_shape[0]
			fil_cols = filter_shape[1]

			filter = tf.get_variable( initializer=tf.truncated_normal(filter_shape, 
									dtype=tf.float32, 
									stddev=stddev),
									name='weights')


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

		exp_dists = tf.exp(-self.distances)
		print("exp_dists shape")
		print(exp_dists.get_shape())

		groundtruths_cmpl = (1-self.groundtruths)		
		print("groundtruths_cmpl shape")
		print(groundtruths_cmpl.get_shape())

		loss = ( (exp_dists * self.output) * groundtruths_cmpl ) + \
			( tf.log(1+tf.exp(-self.output)) * (self.groundtruths + (groundtruths_cmpl*exp_dists) ) )

		return tf.reduce_mean( tf.reduce_sum(loss, axis=[1,2]) )
		
	def build_model(self, name, reuse = False):
		
		self.parameters = []
		with tf.variable_scope(name, reuse=reuse) as scope:

			self.initialize_all_variables()

			images = self.preprocess(name='preprocess')

			conv1_1 = self.conv2d(input = images,  filter_shape = [3, 3, 3,   64],  name = "conv1_1")

			# print("conv1_1 shape")
			# print(conv1_1.get_shape())

			conv1_2 = self.conv2d(input = conv1_1, filter_shape = [3, 3, 64,  64],  name = "conv1_2")
			pool1 = self.max_pool(input = conv1_2, name = "pool1")

			# print("pool1 shape")
			# print(pool1.get_shape())

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

			# print("pool4 shape")
			# print(pool4.get_shape())

			conv5_1 = self.conv2d(input = pool4,   filter_shape = [3, 3, 512, 512], name = "conv5_1")
			conv5_2 = self.conv2d(input = conv5_1, filter_shape = [3, 3, 512, 512], name = "conv5_2")
			conv5_3 = self.conv2d(input = conv5_2, filter_shape = [3, 3, 512, 512], name = "conv5_3")
			pool5 = self.max_pool(input = conv5_3, name = "pool5")

			# print("pool5 shape")
			# print(pool5.get_shape())

			# No padding in c14-18 and DCs
			conv14 = self.conv2d(input = pool5,   filter_shape = [7, 7, 512,  2048], name = "conv14")
			conv15 = self.conv2d(input = conv14,  filter_shape = [1, 1, 2048, 512],  name = "conv15")


			# print("conv14 shape")
			# print(conv14.get_shape())

			# print("conv15 shape")
			# print(conv15.get_shape())

			# print("conv15 shape")
			# print(conv15.get_shape())

			conv16 = self.conv2d(input = conv15,  filter_shape = [1, 1, 512,  1],    name = "conv16")

			# print("conv16 shape")
			# print(conv16.get_shape())
			
			conv17 = self.conv2d(input = pool4,   filter_shape = [1, 1, 512,  1],    name = "conv17")

			# print("conv17 shape")
			# print(conv17.get_shape())

			conv18 = self.conv2d(input = pool3,   filter_shape = [1, 1, 256,  1],    name = "conv18")

			# print("conv18 shape")
			# print(conv18.get_shape())

			# deconv1 = self.deconv2d_custom(conv16, filter_shape=[4, 4], name="deconv_1")

			# print("deconv1 shape")
			# print(deconv1.get_shape())


			deconv1 = self.deconv2d(conv16, filter_shape=[4, 4, 1, 1], output_shape=[self.batch_size, 13, 13, 1], name="deconv_1")

			# print("deconv1 shape")
			# print(deconv1.get_shape())

			fusion1 = self.fusion(deconv1, conv17, name="fusion_1")

			# print("fusion1 shape")
			# print(fusion1.get_shape())

			deconv2 = self.deconv2d(fusion1, filter_shape=[4, 4, 1, 1], output_shape=[self.batch_size, 26, 26, 1], name="deconv_2")

			# print("deconv2 shape")
			# print(deconv2.get_shape())

			fusion2 = self.fusion(deconv2, conv18, name="fusion_2")

			# print("fusion2 shape")
			# print(fusion2.get_shape())

			deconv3 = self.deconv2d(deconv2, filter_shape=[16, 16, 1, 1], output_shape=[self.batch_size, 208, 208, 1],
									strides=(1,8,8,1), name="deconv_3")

			# print("deconv3 shape")
			# print(deconv3.get_shape())

			crop = self.crop(deconv3, name="crop")
			output = tf.reshape(crop , shape=[self.batch_size, self.inp_dim, self.inp_dim] )

			# print("crop shape")
			# print(output.get_shape())
			
			self.output = output

			print("building model done")

	def build_optimizer(self):
		
		self.loss = self.overall_loss()
		print("loss shape")
		print(self.loss.get_shape())

		self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

		self.gradients = self.optimizer.compute_gradients(self.loss)

		self.capped_gradients = [( tf.clip_by_value( grad, -self.max_gradient_norm, self.max_gradient_norm ), variable ) for 
																		grad, variable in self.gradients if grad is not None]

		self.train_op = self.optimizer.apply_gradients(self.capped_gradients)

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

def f_function(self, distance, name = None):
			
	max_dist = math.sqrt(np.amax(distance))
	threshold = 0.3 * max_dist

	distance[distance == 0] = 0
	distance[distance > 0 and distance < threshold] = distance / max_dist
	distance[distance > threshold] = threshold / max_dist

	return distance  


if __name__ == '__main__':

	# test_deconv2d_custom()
	# sys.exit()

	sess = tf.Session()
	
	model = rsrcnn('vgg16_c1-c13_weights', sess)

	print("model creation done!")	

	images = [] 
	for file in listdir(IMAGES_PATH):
		image = ndimage.imread(IMAGES_PATH + file)
		images.append(image)

	groundtruths = []
	for file in listdir(GROUNDTRUTHS_PATH):
		groundtruth = ndimage.imread(GROUNDTRUTHS_PATH + file, mode = 'L')
		groundtruths.append(groundtruth)		

	distances = []
	distances_max = []
	for file in listdir(DISTANCES_PATH):
		distance_image = ndimage.imread(DISTANCES_PATH + file, mode = 'L')
		distances.append(f_function(distance_image))

	# print(len(images))	
	# validation_size = int(len(images) / 10)
	# partitions = [0] * len(images)
	# partitions[:validation_size] = [1] * validation_size
	# random.shuffle(partitions)
	# print(partitions)

	# train_images, validation_images = tf.dynamic_partition(images, partitions, 2)
	# train_groundtruths, validation_groundtruths = tf.dynamic_partition(groundtruths, partitions, 2)	

	# # define tensor shape
	# train_images.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, 3])
	# validation_images.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, 3])
	# train_groundtruths.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, 1])
	# validation_groundtruths.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, 1])

	# train_images_batch, train_groundtruths_batch = tf.train.batch(
 #                                    	[train_images, train_groundtruths],
 #                                    	batch_size=BATCH_SIZE)
	# validation_images_batch, validation_groundtruths_batch = tf.train.batch(
 #                                    	[validation_images, validation_groundtruths],
 #                                    	batch_size=BATCH_SIZE)

	# img1 = misc.imread(IMAGES_PATH + "10078660_15_00000.png", mode='RGB') # example of image
	# img1 = misc.imresize(img1, (200, 200))

	validation_images = images[0:int(len(images) / 10)]
	train_images = images[int(len(images) / 10) :]

	validation_groundtruths = groundtruths[0:int(len(groundtruths) / 10)]
	train_groundtruths = groundtruths[int(len(groundtruths) / 10) :]

	validation_distances = distances[0:int(len(distances) / 10)]
	train_distances = distances[int(len(distances) / 10) :]


	model.sess.run(tf.global_variables_initializer())

	print("All variables initialized.")


	for i in range(int(len(images) / BATCH_SIZE)):
		sess.run(model.output, feed_dict={model.distances : train_distances[i * BATCH_SIZE: (i + 1) * BATCH_SIZE],
										  model.groundtruths : train_groundtruths[i * BATCH_SIZE: (i + 1) * BATCH_SIZE],
										  model.imgs : train_images[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
										  })




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