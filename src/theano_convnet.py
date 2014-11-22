import numpy as np
from operator import mul

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from theano.sandbox.cuda.basic_ops import gpu_contiguous
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
from pylearn2.sandbox.cuda_convnet.pool import MaxPool

from theano_utils import * 

# ---------------------------------------------------------------------
# Implements convolutional network architecture using Theano.
# Based on examples in Theano Deep Learning Tutorials:
# 	http://www.deeplearning.net/tutorial/
# Implements dropout in hidden layers (convolutions don't need it).
# ---------------------------------------------------------------------

class OutputLayer(object):
	'''
	Basically a multiclass logistic regression classifier.
	'''
	def __init__(self, input, n_in, n_out, W=None, b=None):

		if W is None:
			W = theano.shared(value=np.zeros((n_in, n_out), dtype=theano.config.floatX), name='W')
		if b is None:
			b = theano.shared(value=np.zeros((n_out,), dtype=theano.config.floatX), name='b')

		self.W = W
		self.b = b

		# probability of being y given example x: WX + b
		self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

		# prediction is class with highest probability
		self.y_pred = T.argmax(self.p_y_given_x, axis=1)

		# keep track of all weight parameters
		self.params = [self.W, self.b]

	def negative_log_likelihood(self,y):
		return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]),y])

	def errors(self,y):
		return T.sum(T.neq(self.y_pred, y))


class HiddenLayer(object):
	'''
	A simple hidden layer with choice of activation and dropout.
	'''
	def __init__(self, rng, input, n_in, n_out, activation, dropout_rate=0, W=None, b=None):
		self.input = input
		self.activation = activation 

		if W is None:
			W_values = np.asarray(
				rng.uniform(
					low=-np.sqrt(6. / (n_in + n_out)),
					high=np.sqrt(6. / (n_in + n_out)),
					size=(n_in, n_out)
					),
				dtype=theano.config.floatX)

			W = theano.shared(value=W_values, name='W', borrow=True)

		if b is None:
			b_values = np.zeros((n_out,), dtype=theano.config.floatX)
			b = theano.shared(value=b_values, name='b', borrow=True)


		self.W = W 
		self.b = b

		out = activation(T.dot(input, self.W) + self.b)
		self.output = (out if dropout_rate == 0 else dropout(out, dropout_rate, rng))

		self.params = [self.W, self.b]


class ConvLayer(object):
	'''
	A convolutional layer using Theano's built-in 2-D convolution and subsampling.
	'''
	def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
		"""
		Allocate a LeNetConvPoolLayer with shared variable internal parameters.

		:type rng: np.random.RandomState
		:param rng: a random number generator used to initialize weights

		:type input: theano.tensor.dtensor4
		:param input: symbolic image tensor, of shape image_shape

		:type filter_shape: tuple or list of length 4
		:param filter_shape: (number of filters, num input feature maps,
		                      filter height, filter width)

		:type image_shape: tuple or list of length 4
		:param image_shape: (batch size, num input feature maps,
		                     image height, image width)

		:type poolsize: tuple or list of length 2
		:param poolsize: the downsampling (pooling) factor (#rows, #cols)
		"""

		assert image_shape[1] == filter_shape[1]
		self.input = input

		# there are "num input feature maps * filter height * filter width"
		# inputs to each hidden unit
		fan_in = np.prod(filter_shape[1:])
		# each unit in the lower layer receives a gradient from:
		# "num output feature maps * filter height * filter width" /
		#   pooling size
		if poolsize is None:
			fan_out = filter_shape[0] * np.prod(filter_shape[2:])
		else: 
			fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /
		           np.prod(poolsize))
		# initialize weights with random weights
		W_bound = np.sqrt(6. / (fan_in + fan_out))
		self.W = theano.shared(
		    np.asarray(
		        rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
		        dtype=theano.config.floatX
		    ),
		    borrow=True
		)

		# the bias is a 1D tensor -- one bias per output feature map
		b_values = np.ones((filter_shape[0],), dtype=theano.config.floatX)
		self.b = theano.shared(value=b_values, borrow=True)

		# convolve input feature maps with filters
		# conv_out = conv.conv2d(
		#     input=input,
		#     filters=self.W,
		#     filter_shape=filter_shape,
		#     image_shape=image_shape
		# )
		input_shuffled = input.dimshuffle(1, 2, 3, 0) # bc01 to c01b
		filters_shuffled = self.W.dimshuffle(1, 2, 3, 0) # bc01 to c01b
		conv_op = FilterActs(stride=1, partial_sum=1)
		contiguous_input = gpu_contiguous(input_shuffled)
		contiguous_filters = gpu_contiguous(filters_shuffled)
		conv_out_shuffled = conv_op(contiguous_input, contiguous_filters)

		# downsample each feature map individually, using maxpooling
		if poolsize is not None:
			# pooled_out = downsample.max_pool_2d(
			#     input=conv_out,
			#     ds=poolsize,
			#     ignore_border=True
			# )
			pool_op = MaxPool(ds=poolsize[0], stride=poolsize[0])
			pooled_out_shuffled = pool_op(conv_out_shuffled)
			pooled_out = pooled_out_shuffled.dimshuffle(3, 0, 1, 2) # c01b to bc01
		else:
			pooled_out = conv_out_shuffled.dimshuffle(3,0,1,2) #c01b to bc01

		# add the bias term. Since the bias is a vector (1D array), we first
		# reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
		# thus be broadcasted across mini-batches and feature map
		# width & height
		# self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
		self.output = rectified_linear(pooled_out + self.b.dimshuffle('x',0,'x','x'))
		# self.output = T.log(1 + T.exp(pooled_out + self.b.dimshuffle('x',0,'x','x')))

		# store parameters of this layer
		self.params = [self.W, self.b]


class ConvNet(object):
	'''
	A deep convolutional neural network with dropout in hidden layers.
	You can adjust:
	- The number of layers and size of each layer 
	- The presence and value of dropout rate
	- Different activations in layers
	'''
	def __init__(self, rng, 
		conv_filter_shapes, 
		image_shapes, 
		poolsizes, 
		hidden_layer_sizes, 
		n_outputs, 
		learning_rate, 
		dropout_rate, 
		batch_size,
		activations,
		train_set_x,
		train_set_y,
		valid_set_x=None,
		valid_set_y=None,
		test_set=None):


		
		x = T.matrix('hakuna')
		y = T.ivector('matata')

		self.layers = []
		self.dropout_layers = []

		self.learning_rate = theano.shared(np.asarray(learning_rate,
        dtype=theano.config.floatX))

		# don't do dropout on input going into convolution...
		next_layer_input = x.reshape((batch_size, 1, 100,100))
		next_dropout_layer_input = next_layer_input

		# add the convolution layers
		conv_count = 0
		for filter_shape, image_shape in zip(conv_filter_shapes, image_shapes):

			# keep a set for dropout training...
			next_dropout_layer = ConvLayer(rng,
				input=next_dropout_layer_input,
				filter_shape=filter_shape,
				image_shape=image_shape,
				poolsize=poolsizes[conv_count])
			self.dropout_layers.append(next_dropout_layer)
			next_dropout_layer_input = next_dropout_layer.output 

			# for convolutions, non-dropout is the same layer
			next_hidden_layer = next_dropout_layer
			self.layers.append(next_hidden_layer)
			next_layer_input = next_hidden_layer.output

			conv_count += 1

		# prepare dropout on layer input
		next_layer_input = next_layer_input.flatten(2)
		# next_layer_input = next_layer_input.reshape((next_layer_input.shape[0], -1))
		next_dropout_layer_input = dropout(next_layer_input, dropout_rate, rng)

		# add the hidden layers 
		hidden_count = 0
		# number of kernels * new image size
		n_in = conv_filter_shapes[-1][0] * get_final_image_size(conv_filter_shapes, image_shapes, poolsizes[-1])
		for n_out in hidden_layer_sizes:

			# the dropout layers for training...
			next_dropout_layer = HiddenLayer(rng, 
				input=next_dropout_layer_input, 
				n_in=n_in, n_out=n_out, 
				activation=activations[hidden_count], 
				dropout_rate=dropout_rate)
			self.dropout_layers.append(next_dropout_layer)
			next_dropout_layer_input = next_dropout_layer.output

			# corresponding regular layers
			next_hidden_layer = HiddenLayer(rng, 
				input=next_layer_input, 
				n_in=n_in, n_out=n_out, 
				activation=activations[hidden_count], 
				W=next_dropout_layer.W*(1 - dropout_rate), 
				b=next_dropout_layer.b)
			self.layers.append(next_hidden_layer)
			next_layer_input = next_hidden_layer.output

			hidden_count += 1
			n_out = n_in


		n_in = hidden_layer_sizes[-1]
		n_out = n_outputs 
		dropout_output = OutputLayer(input=next_dropout_layer_input, n_in=n_in, n_out=n_out)
		self.dropout_layers.append(dropout_output)

		output_layer = OutputLayer(input=next_dropout_layer_input, n_in=n_in, n_out=n_out, W=dropout_output.W*(1 - dropout_rate), b=dropout_output.b)
		self.layers.append(output_layer)

		# errors for training (dropout) and validation (no dropout)...
		self.dropout_nll = self.dropout_layers[-1].negative_log_likelihood
		self.dropout_errors = self.dropout_layers[-1].errors

		self.nll = self.layers[-1].negative_log_likelihood
		self.errors = self.layers[-1].errors 

		# dropout parameters will be used to calculate gradients during training
		self.params = [ param for layer in self.dropout_layers for param in layer.params ]

		# --------------------------
		# Compile training functions
		# --------------------------

		index = T.lscalar('index')

		# put the data into shared variable so that minibatch access occurs on the gpu
		# ... hopefully it all fits on the gpu or we've got slow-down
		self.train_set_x, self.train_set_y, self.valid_set_x, self.valid_set_y = shared_dataset(train_set_x, train_set_y, valid_set_x, valid_set_y)
		if test_set is not None:
			self.test_set = theano.shared(
				np.asarray(test_set,dtype=theano.config.floatX),
				borrow=True)

		self.validate_model=None
		# compile validation function if you have a validation set
		if valid_set_x is not None and valid_set_y is not None: 
			self.validate_model = theano.function(inputs=[index], 
				outputs=self.errors(y),
				givens={
					x: self.valid_set_x[index * batch_size:(index + 1) * batch_size],
					y: self.valid_set_y[index * batch_size:(index + 1) * batch_size]
				})

		# allows you to get training error on the whole training set
		self.get_train_error = theano.function(inputs=[index],
			outputs=self.errors(y),
			givens={
				x: self.train_set_x[index * batch_size:(index + 1) * batch_size],
				y: self.train_set_y[index * batch_size:(index + 1) * batch_size]
			})

		# Compute gradients
		self.grads = T.grad(self.dropout_nll(y), self.params)

		# SGD weights update
		updates = [(param_i, param_i - learning_rate * grad_i) for param_i, grad_i in zip(self.params, self.grads)]

		# Compile training function that returns training cost, and updates model parameters. 
		train_output, train_errors = self.dropout_nll(y), self.dropout_errors(y)
		self.train_model = theano.function(inputs=[index], 
			outputs=[train_output, train_errors],
			updates=updates,
			givens={
				x: self.train_set_x[index * batch_size:(index + 1) * batch_size],
				y: self.train_set_y[index * batch_size:(index + 1) * batch_size]
			})

		if test_set is not None:
			self.predict_model = theano.function(inputs=[index],
				outputs=self.layers[-1].y_pred,
				givens={
					x: self.test_set[index * batch_size:(index + 1) * batch_size]
				})

		if valid_set_x is not None:
			self.predict_valid = theano.function(inputs=[index],
				outputs=self.layers[-1].y_pred,
				givens={
					x: self.valid_set_x[index * batch_size:(index + 1) * batch_size]
				})

		self.decay_learning_rate = theano.function(inputs=[], outputs=self.learning_rate,
            updates={self.learning_rate: self.learning_rate * 0.995})






