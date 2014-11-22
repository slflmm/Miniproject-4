import numpy as np

import theano
import theano.tensor as T

def dropout(layer_output, dropout_rate, rng):
	'''
	Applies dropout on the provided layer output.
	'''
	srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
	mask = srng.binomial(n=1, p=1-dropout_rate, size=layer_output.shape)
	return layer_output * T.cast(mask, theano.config.floatX)

def rectified_linear(x):
	'''
	Rectified linear unit activation. Seems to work very well in deep nets.
	'''
	return T.switch(x >= 0, x, 0)#T.maximum(0.0, x)

def sigmoid(x):
	'''
	Canonical sigmoid activation.
	'''
	return T.nnet.sigmoid(x)

def tanh(x):
	'''
	Activation. A lot like sigmoid.
	'''
	return T.tanh(x)

def shared_dataset(train_x, train_y, valid_x=None, valid_y=None):
	# print np.asarray(train_x,dtype=theano.config.floatX).reshape(10000,1,48,48).shape
	# temp = np.asarray(train_x,dtype=theano.config.floatX).reshape(len(train_x),1,48,48)
	temp = np.asarray(train_x,dtype=theano.config.floatX)
	shared_train_x = theano.shared(temp, borrow=True)
	temp = np.asarray(train_y,dtype=theano.config.floatX)
	shared_train_y = T.cast(theano.shared(temp, borrow=True), 'int32')

	if valid_x is not None:
		temp = np.asarray(valid_x, dtype=theano.config.floatX)#.reshape(len(valid_x),1,48,48)
		shared_valid_x = theano.shared(temp, borrow=True)
		shared_valid_y = T.cast(theano.shared(np.asarray(valid_y, dtype=theano.config.floatX), borrow=True), 'int32')

	return shared_train_x, shared_train_y, shared_valid_x, shared_valid_y

def get_final_image_size(filter_shapes, image_shapes, pool):
	if pool is not None:
		current = (image_shapes[-1][2] - filter_shapes[-1][2] + 1)/2
	else:
		current = image_shapes[-1][2] - filter_shapes[-1][2] + 1
	return current*current

def gradient_updates_momentum(cost, params, learning_rate, momentum):
    # List of update steps for each parameter
    updates = []
    # Just gradient descent on cost
    for param in params:
        # For each parameter, we'll create a param_update shared variable.
        # This variable will keep track of the parameter's update step across iterations.
        # We initialize it to 0
        param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
        # Each parameter is updated by taking a step in the direction of the gradient.
        # However, we also "mix in" the previous step according to the given momentum value.
        # Note that when updating param_update, we are using its old value and also the new gradient step.
        updates.append((param, param - learning_rate*param_update))
        # Note that we don't need to derive backpropagation to compute updates - just use T.grad!
        updates.append((param_update, momentum*param_update + (1. - momentum)*T.grad(cost, param)))
    return updates

