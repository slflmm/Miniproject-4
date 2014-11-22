import urllib
import random
from utils import *
import os

N_SAMPLES = 8000
KEY = 'AIzaSyBpRsrt-Kk5PupV9gDq8vXiDQeHpnkdOhw'
PATH = '/Users/stephanielaflamme/Desktop/Miniproject-4/raw_images/'

COORDINATES = [
[(45.4965279,-73.581726), (45.5055724,-73.5729745), (45.5189297,-73.5664739), (45.5144276,-73.5566828), (45.4937764,-73.5749687)],
	[(45.4998343,-73.5561395), (45.501248,-73.5602581), (45.5094521,-73.554103), (45.5113367,-73.5518386), (45.5110466,-73.5501327), (45.5033755,-73.5539042)],
	[(45.508741,-73.5613417), (45.5069724,-73.5610176), (45.5063975,-73.5596936), (45.5074945,-73.5586321)]
	# [(45.5226378,-73.5524594), (45.5225347,-73.5522357),(45.5154124,-73.5588154), (45.5154805,-73.5589622)],
	# [(45.5083689,-73.571288), (45.523683,-73.605523), (45.5394978,-73.581831), (45.5384869,-73.5600707)],
	# [(45.5100436,-73.6116584), (45.5170896,-73.6265568), (45.5243897,-73.6113895), (45.5168201,-73.5935993),(45.5102934,-73.6002257)],
	# [(45.4758723,-73.6021969), (45.4880136,-73.5830917), (45.4895681,-73.607528), (45.4907769,-73.615131), (45.4800412,-73.6114921)],
	# [(45.5377389,-73.5421315), (45.5424521,-73.5579556), (45.5582896,-73.5487811), (45.551642,-73.533546)],
	# [(45.583204,-73.6523514), (45.5777561,-73.6404374), (45.610674,-73.6078028), (45.6258078,-73.6239303)]
	]

NEIGHBOURHOODS = ['Downtown', 'Old Montreal', 'Chinatown', 'Gay Village', 'Plateau', 'Outremont', 'Westmount', 'Hochelaga', 'Montreal-Nord']

# for neighbourhood, coordinates in NEIGHBOURHOODS:
for neighbourhood_id, coordinates in enumerate(COORDINATES):
	save_location = PATH + NEIGHBOURHOODS[neighbourhood_id] + '/'
	n_images = 0
	if neighbourhood_id == 0 or neighbourhood_id == 1:
		continue
	if neighbourhood_id == 2:
		n_images = 7054
	max_lat, min_lat = max(coordinates,key=lambda item:item[0])[0], min(coordinates,key=lambda item:item[0])[0]
	max_long, min_long = max(coordinates, key=lambda item:item[1])[1], min(coordinates, key=lambda item:item[1])[1]
	while n_images < N_SAMPLES:
		r_lat = random.uniform(min_lat, max_lat)
		r_long = random.uniform(min_long, max_long)
		r_orient = random.uniform(0,360) # any camera orientation (in the NSEW dimension)
		r_height = random.uniform(0,35) # angle of camera relative to ground
		if point_in_poly(r_lat, r_long,coordinates):
			urllib.urlretrieve("http://maps.googleapis.com/maps/api/streetview?size=400x400&location=%f,%f&heading=%f&fov=100&pitch=%f&key=%s" \
				% (r_lat, r_long, r_orient, r_height, KEY), \
				filename=save_location + "%d.png" % n_images)
			# discard fake images...
			if is_fake(save_location + "%d.png" % n_images):
				os.remove(save_location + "%d.png" % n_images)
				continue
			n_images +=1

# old montreal 7205, 7359, 7553, 7752
# gay village examples: 67, 128