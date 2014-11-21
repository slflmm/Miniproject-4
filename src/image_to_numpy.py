import scipy
from random import shuffle

PATH = '/Users/stephanielaflamme/Desktop/Miniproject-4/raw_images/'
DATASET_PATH = '/Users/stephanielaflamme/Desktop/Miniproject-4/numpy_data/dataset'

NEIGHBOURHOODS = ['Downtown', 'Old Montreal', 'Chinatown', 'Gay Village', 'Plateau', 'Outremont', 'Westmount', 'Hochelaga', 'Montreal-Nord']

dataset = []

for idx, neighbourhood in enumerate(NEIGHBOURHOODS):
	neighbourhood_path = PATH + neighbourhood + '/'
	for i in xrange(8000):
		image_path = neighbourhood_path + "%d.png" % i
		im = scipy.misc.imread(image_path, flatten=1)
		dataset.append([im, idx])

shuffle(dataset)
np.save(DATASET_PATH, dataset)