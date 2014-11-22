import numpy as np
import random

from utils import * 


class HandmadeClassifier(object):
	'''
	Template for a classifier.
	'''
	def __init__(self):
		pass

	def train(self, examples, outputs):
		pass

	def predict(self, example):
		pass 

class Perceptron(HandmadeClassifier):
	'''
	A simple multiclass perceptron.
	'''
	def __init__(self, alpha=0.1, activation=lambda x: step(x), n_iter=10):
		self.alpha = alpha
		self.activation = activation
		self.n_iter = n_iter

	def train(self, examples, outputs):
		'''
		Trains a perceptron.
		Expects arrays of examples and outputs (as one-hot vectors).
		'''
		n_examples = examples.shape[0]
		n_features = examples.shape[1]
		n_outputs = outputs.shape[1]

		self.w = np.random.uniform(-0.5,0.5,(n_features + 1, n_outputs))
		bias_examples = np.ones((n_examples, n_features + 1))
		bias_examples[:,1:] = examples

		# until max iterations are reached or data is completely classified...
		for _ in range(self.n_iter):
			iter_error = 0
			for x, y in zip(bias_examples, outputs):
				# get output and error
				out = self.activation(np.dot(self.w.T, x))
				error = y - out
				# if you classified wrong:
				if np.sum(error) != 0:
					iter_error += 1
					# modify the relevant weights
					self.w[:,np.argmax(out)] -= self.alpha * x
					self.w[:,np.argmax(y)] += self.alpha * x
			if (iter_error < 1) : break

	def predict(self, example):
		'''
		Predicts the output for example.
		'''
		n_features = example.size
		bias_example = np.ones(n_features + 1)
		bias_example[1:] = example
		return np.argmax(self.activation(np.dot(self.w.T, bias_example)))

class NeuralNet(HandmadeClassifier):
	'''
	A simple fully-connected neural network.
	'''
	def __init__(self,layers,alpha=0.01, epochs=10):
		self.activation = sigmoid
		self.activation_prime = sigmoid_prime

		self.alpha = alpha

		self.epochs = epochs

		self.weights = []

		# make the weights...
		for i in range(1, len(layers) - 1):
			r = 2*np.random.random((layers[i-1] + 1, layers[i] + 1)) - 1
			self.weights.append(r)
		r = 2*np.random.random((layers[i] + 1, layers[i+1])) - 1
		self.weights.append(r)

	def train(self, examples, outputs):
		'''
		Performs SGD on minibatches.
		'''
		ones = np.atleast_2d(np.ones(examples.shape[0]))
		examples = np.concatenate((ones.T, examples),axis=1)

		for k in range(self.epochs):
			for i, ex in enumerate(examples): 
				
				# get feedforward activations of layers
				a = [ex]
				for l in range(len(self.weights)):
					dot_value = np.dot(a[l], self.weights[l])
					activation = self.activation(dot_value)
					a.append(activation)

				# get error, list deltas going backward through network
				error = outputs[i] - a[-1]
				deltas = [error*self.activation_prime(a[-1])]
				for l in range(len(a) - 2, 0, -1):
					deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_prime(a[l]))
				deltas.reverse()

				# update the weights
				for i in range(len(self.weights)):
					layer = np.atleast_2d(a[i])
					delta = np.atleast_2d(deltas[i])
					self.weights[i] += self.alpha * layer.T.dot(delta)
			print 'Finished epoch %d' % (k+1)

	def predict(self, example):
		'''
		Returns the most likely label for the example. 
		'''
		a = np.concatenate((np.ones(1).T, np.array(example)))
		for l in range(0, len(self.weights)):
			a = self.activation(np.dot(a, self.weights[l]))
		return np.argmax(a)

