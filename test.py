import urllib
import random

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

'''
This uses the boundaries of Chinatown to generate 40 images within the neighbourhood.
We should get more images than this...
ALSO, the key AIzaSyBpRsrt-Kk5PupV9gDq8vXiDQeHpnkdOhw allows us to download 25000 images daily.
If we each get our own key, we can get 75000 images daily--and our whole dataset pretty fast.
BEFORE WE COLLECT DATA, we still need to decide:
(1) How big our dataset should be 
(2) Exactly which neighborhoods we're using
(3) Using (1) and (2), how many images per neighbourhood
'''
n_images = 1
chinatown_poly = [(45.508741,-73.5613417), (45.5069724,-73.5610176), (45.5063975,-73.5596936), (45.5074945,-73.5586321)]
while n_images <= 40:
	r_long = random.uniform(45.5063975, 45.508741) # gets first number between max and min boundaries
	r_lat = random.uniform(-73.5613417,-73.5586321) # gets second number between max and min boundaries
	r_orient = random.uniform(0,360) # any camera orientation (in the NSEW dimension)
	r_height = random.uniform(0,35) # angle of camera relative to ground
	if point_in_poly(r_long, r_lat, chinatown_poly):
		urllib.urlretrieve("http://maps.googleapis.com/maps/api/streetview?size=400x400&location=%f,%f&heading=%f&fov=100&pitch=%f&key=AIzaSyBpRsrt-Kk5PupV9gDq8vXiDQeHpnkdOhw" % (r_long, r_lat, r_orient, r_height), filename="img%d.png" %n_images)
		n_images +=1




