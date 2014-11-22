import numpy as np
import time


import theano
import theano.tensor as T

from theano_utils import * 

# ---------------------------------------------------------------------
# Training, validating, and predicting with a convnet.
# Based on examples in Theano Deep Learning Tutorials:
# 	http://www.deeplearning.net/tutorial/
# TODO: prettify?
# ---------------------------------------------------------------------

class Trainer(object):
	'''
	Training and validation of a neural net.
	Also gives predictions.
	'''

	def __init__(self, neural_network):
		self.classifier = neural_network

	def train(self, 
		learning_rate, 
		n_epochs, 
		batch_size):
		'''
		Compiles functions for training, then trains.
		Learns by doing SGD on minibatches.
		Returns average training cost and average validation cost of final model if a validation set is provided.
		'''

		n_train_batches = self.classifier.train_set_x.get_value(borrow=True).shape[0] / batch_size
		n_valid_batches = self.classifier.valid_set_x.get_value(borrow=True).shape[0] / batch_size
		n_test_batches = self.classifier.test_set.get_value(borrow=True).shape[0]/batch_size
		

		# Then do the training and validation!
		training_error = 0
		epoch = 0

		best_val_error = 1
		best_predict = []
		best_val_predict = []

		start_time = time.clock()

		while (epoch < n_epochs):
			epoch = epoch + 1

			# train on all examples in minibatches 
			# if you're on the last epoch, track your average error
			for minibatch_index in xrange(n_train_batches):
				self.classifier.train_model(minibatch_index)

			# print 'Completed epoch %d. Code has run for %.2fm.' %(epoch, (time.clock() - start_time)/60)

			if self.classifier.valid_set_x is not None and epoch%10==0:
				print 'EPOCH %d:' % epoch

				training_errors = [self.classifier.get_train_error(i) for i in xrange(n_train_batches)]
				train_error = np.mean(training_errors)/batch_size
				print 'Training error is %f' % train_error

				validation_errors = [self.classifier.validate_model(i) for i in xrange(n_valid_batches)]
				val_error =  np.mean(validation_errors)/batch_size
				print 'Validation error is %f' % val_error

				if val_error < best_val_error:
					best_val_error = val_error
					best_val_predict = [self.classifier.predict_valid(i) for i in xrange(n_valid_batches)]
					best_predict = [self.classifier.predict_model(i) for i in xrange(n_test_batches)]

			self.classifier.decay_learning_rate()

		end_time = time.clock()

		print 'Finished training!\n The code ran for %.2fm.' % ((end_time - start_time) / 60.)

		return best_val_error, best_val_predict, best_predict





