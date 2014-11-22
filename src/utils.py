import Image
import numpy as np

def point_in_poly(x,y,poly):
	'''
	Code obtained from http://geospatialpython.com/2011/01/point-in-polygon.html
	'''
	n = len(poly)
	inside = False

	p1x,p1y = poly[0]
	for i in range(n+1):
	    p2x,p2y = poly[i % n]
	    if y > min(p1y,p2y):
	        if y <= max(p1y,p2y):
	            if x <= max(p1x,p2x):
	                if p1y != p2y:
	                    xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
	                if p1x == p2x or x <= xints:
	                    inside = not inside
	    p1x,p1y = p2x,p2y

	return inside


def is_fake(img_path):
    im = Image.open(img_path).convert('RGB')
    w,h = im.size
    red, green, blue = [], [], []
    for i in range(w):
        r,g,b = im.getpixel((i,5))
        if r!=228 or g!=227 or b!=223:
        	return False
    return True

def one_hot_vectorizer(n):
	v = np.zeros(10)
	v[n] = 1
	return v

def step(x):
	'''
	Just the step function.
	Works for numbers and arrays.
	'''
	return np.sign(x)

class CrossValidation(object):
    '''
    Iterator that returns 1/k of the data as validation data and
    the rest as training data, for every of the k pieces.
    '''
    def __init__(self, examples, outputs, k=10):
        assert len(examples) == len(outputs)

        self.examples = examples
        self.outputs = outputs
        self.k = len(outputs) // k
        self.i = 0

    def __iter__(self):
        return self

    def next(self):
        s, e = self.i * self.k, (self.i + 1) * self.k
        if s >= len(self.examples):
            raise StopIteration
        self.i += 1

        train_data = np.concatenate((self.examples[:s,:],self.examples[e:,:]))
        train_result = np.concatenate((self.outputs[:s],self.outputs[e:]))

        test_data = self.examples[s:e,:]
        test_result = self.outputs[s:e]

        return train_data, train_result, test_data, test_result