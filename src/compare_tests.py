import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def get_confusion_matrix(Y1, Y2):    
    """
    Generate the confusion matrix between
    the expected outputs and the classifier outputs.    
    Y1: 1D array of expected outputs.
    Y2: 1D array of classifier outputs.
    """    
    # Compute confusion matrix
    matrix = confusion_matrix(Y1, Y2)
    
    print(matrix)
    
    # Show confusion matrix in a separate window.
    plt.matshow(matrix)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

print "Loading test output..."
real_test_outputs = np.load("/home/ml/slafla2/Miniproject-4/numpy_data/test_y.npy")

print "Loading test output..."
test_outputs = np.load("/home/ml/slafla2/Miniproject-4/results/autoencoder/experiment_predictions-11.npy")

error_count = 0.
test_len = len(test_outputs)

for i in range(test_len):
	if test_outputs[i] != real_test_outputs[i]:
		error_count += 1.

print "Error rate:", error_count/test_len
get_confusion_matrix(real_test_outputs[:test_len],test_outputs)

