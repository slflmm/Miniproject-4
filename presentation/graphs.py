import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pylab import *

def get_confusion_matrix(actual, predicted):
    '''
    Returns the confusion matrix
    '''
    m = np.zeros((9,9))
    for a, b in zip(actual, predicted):
    	m[a,b] += 1

    class_totals = np.sum(m, axis=1)

    for i in xrange(9):
    	m[i] = m[i]*1. / class_totals[i]*1.

    return m

def convnet_visualize():

	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)

	epochs = range(10,210,10)

	train = [0.823681, 0.760434, 0.684792, 0.678941, 0.562292, 0.505278, 0.478264, 0.430920, 0.406875, 0.394323, 0.367622, 0.358351, 0.342708, 0.335920, 0.317101, 0.317760, 0.283090, 0.286962, 0.280365, 0.257413]
	valid = [0.827009, 0.764648, 0.693220, 0.690430, 0.579799, 0.526088, 0.505859, 0.472656, 0.453125, 0.458287, 0.448242, 0.438756, 0.439314, 0.444336, 0.433175, 0.450056, 0.428153, 0.433873, 0.446987, 0.433315]

	c1, = ax.plot(epochs, train, marker='D', color='red', label='Training')
	c2, = ax.plot(epochs, valid, marker='D', color='green', label='Validation')

	handles, labels = ax.get_legend_handles_labels()
	ax.legend(handles, labels, loc=0)

	ax.yaxis.set_ticks([0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1])
	ax.yaxis.grid(b=True, which='both', color='black', linestyle='--')
	# labels
	ax.set_xlabel('Number of epochs')
	ax.set_ylabel('Error')

	plt.savefig('/Users/stephanielaflamme/Desktop/Miniproject-4/presentation/best_convnet_learning.pdf')


def convnet_confusion():

	predictions = np.load('/Users/stephanielaflamme/Desktop/Miniproject-4/results/convnet/experiment_predictions-4.npy')
	truth = np.load('/Users/stephanielaflamme/Desktop/Miniproject-4/numpy_data/test_y.npy')

	fig = plt.figure()
	ax = fig.add_subplot(111)
	# make the matrix...
	mat = get_confusion_matrix(truth, predictions).tolist()

	cax = ax.matshow(mat, cmap=cm.jet)
	fig.colorbar(cax)

	for x in xrange(9):
		for y in xrange(9):
			ax.annotate('%4.2f' % (mat[x][y]), xy=(y,x), horizontalalignment='center', verticalalignment='center', color='white')

	plt.xticks(np.arange(9))
	plt.yticks(np.arange(9))
	ax.set_title('Prediction', fontsize=16)
	ax.set_ylabel('True label', fontsize=16)

	plt.savefig('/Users/stephanielaflamme/Desktop/Miniproject-4/presentation/convnet_confusion.pdf')

def autoencode_visualize():

	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)

	epochs = range(1,101)

	print len(epochs)
	train = map(lambda x: x/100, [74.383681, 71.895833, 70.347222, 69.151042, 68.065972, 67.244792, 66.394097, 65.571181, 
		64.842014, 64.102431, 63.460069, 62.744792, 62.079861, 61.314236, 60.630208, 59.899306, 59.104167,
		58.376736, 57.572917, 56.862847, 56.052083, 55.413194, 54.682292, 53.977431, 53.381944, 52.932292,
		52.550347, 53.579861, 52.295139, 53.305556, 53.074653, 51.821181, 51.847222, 53.546875, 54.059028,
		52.786458, 48.585069, 51.786458, 51.805556, 49.069444, 49.786458, 47.430556, 48.704861, 46.107639,
		48.064236, 44.359375, 44.319444, 43.822917, 45.729167, 41.760417, 42.357639, 38.689236, 38.232639,
		41.154514, 39.541667, 40.550347, 33.927083, 39.812500, 36.446181, 36.019097, 33.895833, 32.236111, 
		28.630208, 27.989583, 27.394097, 25.065972, 30.873264, 32.690972, 28.630208, 22.098958, 25.609375,
		26.218750, 23.423611, 22.434028, 23.361111, 19.760417, 23.048611, 28.321181, 23.236111, 22.409722,
		19.961806, 17.222222, 13.571181, 12.156250, 14.223958, 15.180556, 12.534722, 14.423611, 19.501736,
		9.411458, 15.322917, 8.776042, 10.008681, 8.003472, 12.034722, 7.869792, 6.828125, 18.286458,
		4.727431, 4.279514])

	print len(train)

	valid = map(lambda x: x/100, [75.641741, 73.688616, 72.558594, 71.386719, 70.410156, 70.019531, 69.587054, 69.154576,
		68.945312, 68.415179, 68.122210, 67.954799, 67.787388, 67.745536, 67.480469, 67.327009, 67.075893,
		66.824777, 66.727121, 66.434152, 66.531808, 66.322545, 66.406250, 66.378348, 66.210938, 66.657366,
		66.699219, 68.080357, 67.396763, 68.610491, 68.763951, 68.289621, 68.233817, 69.698661, 70.661272,
		69.503348, 67.452567, 70.200893, 70.326451,68.470982, 70.172991, 68.582589, 69.559152, 68.289621,
		69.238281, 68.387277, 68.498884, 68.484933, 70.089286, 68.094308, 67.271205, 67.020089, 66.503906,
		69.545201,68.722098, 69.949777, 66.699219, 69.991629, 68.638393,67.773438, 66.448103, 67.968750,
		66.210938, 67.271205, 65.917969, 66.601562, 68.512835, 70.452009, 68.219866, 67.578125, 67.354911,
		68.415179, 66.866629, 69.126674, 68.680246, 65.862165, 68.791853,71.261161, 67.494420,67.857143,
		66.810826, 67.968750, 65.192522, 65.164621, 67.522321, 67.117746, 66.782924, 68.191964, 68.931362,
		64.606585, 67.075893, 66.336496, 65.053013, 66.545759, 67.633929, 65.876116, 64.815848, 67.703683,
		65.541295, 65.597098])

	print len(valid)

	c1, = ax.plot(epochs, train, marker='D', color='red', label='Training')
	c2, = ax.plot(epochs, valid, marker='D', color='green', label='Validation')

	handles, labels = ax.get_legend_handles_labels()
	ax.legend(handles, labels, loc=0)

	ax.yaxis.set_ticks([0.0, 0.05, 0.1, 0.15, 0.2,0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1])
	ax.yaxis.grid(b=True, which='both', color='black', linestyle='--')
	# labels
	ax.set_xlabel('Number of epochs')
	ax.set_ylabel('Error')

	plt.savefig('/Users/stephanielaflamme/Desktop/Miniproject-4/presentation/best_autoencoder_learning.pdf')

# convnet_visualize()
# convnet_confusion()
autoencode_visualize()


