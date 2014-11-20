import Image

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
