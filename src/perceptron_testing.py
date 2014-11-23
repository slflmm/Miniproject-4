from basic_classifiers import *
from utils import *

import numpy as np

# # ------------------------
# # Loading data
# # ------------------------
print "Loading data..."
test_X = np.load('/Users/ssruan/courses/comp598/finalproject/Archive/test_X.npy')
#test_X = test_X.reshape((len(test_X),200*200))
test_y = np.load('/Users/ssruan/courses/comp598/finalproject/Archive/test_y.npy')

train_X = np.load('/Users/ssruan/courses/comp598/finalproject/Archive/train_X.npy')
#train_X = train_X.reshape((len(train_X),200*200))
train_y = np.load('/Users/ssruan/courses/comp598/finalproject/Archive/train_y.npy')


# # ------------------------
# # Getting test predictions
# # ------------------------
# classifier = Perceptron(alpha=0.01, n_iter=5)
# # train on the entire training set
# print "Training..."
# classifier.train(train_X, np.asarray(map(one_hot_vectorizer, train_y)))

# print "Testing..."
# predictions = map(classifier.predict, test_X)
# testing_correct = filter(lambda x: x[0] == x[1], zip(predictions, test_y))
# print "Obtained accuracy: "
# print len(testing_correct)*1. / len(test_y)

# ---------------------------
# GRIDSEARCH CROSS-VALIDATION
# ---------------------------

print 'Beginning gridsearch...'

# the parameter values under consideration
alphas = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
n_iters = [10, 15, 20, 25, 30, 35]

# this is where we'll save the cross-validation results
cross_val_results = []
cross_val_train_results = []
cross_val_confusion_matrices = []

for n_iter in n_iters:
	for alpha in alphas:
		predictions = []
		success_rates = []
		train_success_rates = []
		# do coss-validation with current parameters
		for data in CrossValidation(examples, categories, k=5):
		    train_data, train_result, valid_data, valid_result = data

		    classifier = Perceptron(alpha=alpha, n_iter=n_iter)
		    # train with one-hot outputs...
		    classifier.train(train_data, np.asarray(map(one_hot_vectorizer, train_result)))

		    training_guesses = map(classifier.predict, train_data)
		    training_correct = filter(lambda x: x[0] == x[1], zip(training_guesses, train_result))
		    training_ratio = len(training_correct)*1. / len(train_result)
		    train_success_rates.append(training_ratio)

		    guesses = map(classifier.predict, valid_data)
		    correct = filter(lambda x: x[0] == x[1], zip(guesses, valid_result))
		    ratio = len(correct)*1. / len(valid_result)
		    success_rates.append(ratio)


		# get the interesting results for this parameter configuration
		train_success_ratio = sum(train_success_rates) / len(train_success_rates) * 100
		success_ratio = sum(success_rates) / len(success_rates) * 100

		print 'Cross-val accuracy for alpha=%f, n_iter=%d: %f' % (alpha, n_iter, success_ratio)

		cross_val_results.append(success_ratio)
		cross_val_train_results.append(train_success_ratio)

# save all the interesting results
np.save('/Users/ssruan/courses/comp598/finalproject/Miniproject-4/results/perceptron/crossval_results', cross_val_results)
np.save('/Users/ssruan/courses/comp598/finalproject/Miniproject-4/results/perceptron/crossval_training_accuracy', cross_val_train_results)






