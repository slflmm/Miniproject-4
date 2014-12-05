from theano_classifier import *
from theano_convnet import *
from theano_utils import *
from utils import *
import pickle

# -----------------
# SETUP
# -----------------
experiment = 15
n_epochs = 10
batch_size = 256
activation = 'ReLU'
filter_shapes =[(32, 1, 9, 9), (64, 32, 6, 6), (80, 64, 5, 5), (80, 80, 5, 5)]
image_shapes = [(256, 1, 100, 100), (256, 32, 23, 23), (256, 64, 9, 9)]
downsampling = [(4,4),(2,2), None, None]
hidden_layers = [500, 500]
learning_rate = 0.05
params = [experiment, batch_size, learning_rate, activation, filter_shapes, image_shapes, downsampling, hidden_layers]
param_titles = ["Experiment number: ", "Batch size: ", "Learning rate: ", "Activation: ", "Filter shapes: ", "Image shapes: ", "Downsampling: ", "Hidden layers: "]

# ------------------------
# Loading data
# ------------------------

print "Loading train input..."
examples = np.load("/home/ml/slafla2/Miniproject-4/numpy_data/contrast_train_X.npy")

print "Loading train output..."
categories = np.load("/home/ml/slafla2/Miniproject-4/numpy_data/train_y.npy")

print "Loading test input..."
test_examples = np.load("/home/ml/slafla2/Miniproject-4/numpy_data/contrast_test_X.npy")

# -----------------
# VALIDATION
# ------------
print "Starting validation..."

train_data, train_result = examples[7200:,:], categories[7200:]
valid_data, valid_result = examples[:7200,:], categories[:7200]
pad_test = np.zeros((7424, test_examples.shape[1]))
pad_test[:7200] = test_examples

print 'Building convnet...'
net = ConvNet(rng = np.random.RandomState(1234),
	# we're getting 720 instead of 320, why?  
	# next image shape is (previous_image_shape - filter_size + 1) / poolsize
	# after  (20,1,7,7) images are (48-7+1 = 42) --> 21 x 21, then (21-6+1 = 16) --> 8x8 
	# after (20, 1, 5, 5) images are (48-5+1 = 44) --> 22 x 22, then (22-5+1 = 18) --> 9x9, then... 
	# (48-9+1=40) => 20x20, then (20-5+1 = 16)=> 8, then (8-5+1=4)=> 2
	# (48-7+1 = 42) => 21x21, then (21-6+1=16)=> 8x8, then (8-4+1=5)=> 5x5, and finally (5-3+1)=> 3x3
	# 21x21, then 16x16, (16-5+1=12) 12x12, (12-5+1=8)
	conv_filter_shapes = filter_shapes,#, [96, 80, 3, 3]], #(22, 22) output, shape ()
	image_shapes = image_shapes,#, (batch_size, 80, 5, 5)], # (9, 9) output, shape (20,50,22,22) #80*2*2=320 but not getting that
	poolsizes=downsampling,
	hidden_layer_sizes=hidden_layers,
	n_outputs=9,
	learning_rate=learning_rate,
	dropout_rate=0.5,
	activations=[rectified_linear]*len(hidden_layers),
	batch_size=batch_size,
	train_set_x=train_data,
	train_set_y=train_result,
	valid_set_x=valid_data,
	valid_set_y=valid_result,
	test_set = pad_test
	)
print 'Making the trainer...'
learner = Trainer(net)

print 'Training...'
best_val, best_val_pred, best_pred = learner.train(learning_rate,n_epochs,batch_size)

print "Best validation error: %f" % best_val

np.save('/home/ml/slafla2/Miniproject-4/results/convnet/experiment_predictions-%d' % experiment, np.asarray(best_pred).flatten())

ws = pickle.load('/home/ml/slafla2/Miniproject-4/results/convnet/weights_.pkl')

for i,w in enumerate(ws):
	if w.ndim == 4:
		image = numpy.hstack(numpy.hstack(w))
		scipy.misc.imsave('kernel_%d.png'%(i), w)

f = open('/home/ml/slafla2/Miniproject-4/results/convnet/experiment_descriptions', 'a')
for param_title, param in zip(param_titles, params):
	f.write(param_title)
	f.write(str(param) + "\n")
f.write("\n\n")
f.close()
