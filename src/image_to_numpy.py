import scipy
import scipy.misc
import numpy as np
from random import shuffle
import Image

PATH = '/Users/stephanielaflamme/Desktop/Miniproject-4/raw_images/'
DATASET_PATH = '/Users/stephanielaflamme/Desktop/Miniproject-4/numpy_data/'

NEIGHBOURHOODS = ['Downtown', 'Old Montreal', 'Chinatown', 'Gay Village', 'Plateau', 'Outremont', 'Westmount', 'Hochelaga', 'Montreal-Nord']

examples = []
labels = []

for idx, neighbourhood in enumerate(NEIGHBOURHOODS):
	neighbourhood_path = PATH + neighbourhood + '/'
	for i in xrange(8000):
		image_path = neighbourhood_path + "%d.png" % i
		im = scipy.misc.imread(image_path, flatten=1)
		im = im.astype('uint8')
		im = scipy.misc.imresize(im, (200,200))
		examples.append(im)
		labels.append(idx)
	print "Finished neighbourhood %d" % idx

combined = zip(examples, labels)
shuffle(combined)

examples[:], labels[:] = zip(*combined)

test_X = examples[:7200]
test_y = labels[:7200]

train_X = examples[7200:]
train_y = labels[7200:]

np.save('/Users/stephanielaflamme/Desktop/Miniproject-4/numpy_data/test_X.npy', test_X)
np.save('/Users/stephanielaflamme/Desktop/Miniproject-4/numpy_data/test_y.npy', test_y)
np.save('/Users/stephanielaflamme/Desktop/Miniproject-4/numpy_data/train_X.npy', train_X)
np.save('/Users/stephanielaflamme/Desktop/Miniproject-4/numpy_data/train_y.npy', train_y)