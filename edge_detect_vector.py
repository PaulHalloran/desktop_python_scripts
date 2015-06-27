import ogr, gdal, osr, os
import numpy as np
import itertools
from math import sqrt,ceil
import cv2
import numpy as np
from osgeo import gdal, ogr
import matplotlib.pyplot as plt

def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    # print 'x = %d, y = %d'%(
    #     ix, iy)
    # assign global variable to access outside of function
    global coords
    coords.append((ix, iy))
    # Disconnect after 2 clicks
    if len(coords) == 2:
        fig.canvas.mpl_disconnect(cid)
        plt.close(1)
    return

####################################
#   MAIN program                  #
####################################

# Get user supplied values
directory = '/home/ph290/Documents/open_cv_stuff/'
imagePath = directory+'tree-rings-0019_web.jpg'

# Read the image
image = cv2.imread(imagePath)
display_image = image.copy()
#blur the image to try and get rid of some of the fine detail
kernel = np.ones((3,3),np.float32)/8
image = cv2.filter2D(image,-1,kernel)

#convert to gray scale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#find the edges
edges = cv2.Canny(gray,200,500)
#NOTE, we coudl probably automate the selectoin of these values by throwing in random numbers to this and teh bluring, and testing how many and how long (on average) the contours are.

#Identify the individual contours. Each contour now represents a single continuous line
contours, hierarchy = cv2.findContours(edges.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


no_lines = np.size(contours)

for i in range(no_lines):
	cv2.drawContours(image, [contours[i]], -1, (255, i, 0), 1)

#cv2.imshow("Lines picked out", image)
#cv2.waitKey(0)

coords = []

image_size = np.shape(image)

plt.close('all')
fig = plt.figure(1)
plt.imshow(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
for i in range(np.size(contours)):
	x = contours[i]
	y = np.reshape(x,[np.shape(x)[0],2])
	plt.plot(y[:,0],y[:,1])
	# Call click func
	cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.xlim([0,image_size[1]])
plt.ylim([0,image_size[0]])
plt.show(1)

print 'select start and end points on figure'


x0 = np.round(coords[0][0])
y0 = np.round(coords[0][1])
x1 = np.round(coords[1][0])
y1 = np.round(coords[1][1])

m = (y0 - y1) / (x0 - x1)
c = y1 - x1  * m


if int(x1) < int(x0):
    xs = np.arange(int(x1),int(x0))
if int(x1) > int(x0):
    xs = np.arange(int(x0),int(x1))

ys = m * xs + c

locations_x = []
locations_y = []

plt.close()
plt.figure(1)
plt.subplot(2, 1, 1)
plt.imshow(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
for i in range(np.size(contours)):
	x = contours[i]
	y = np.reshape(x,[np.shape(x)[0],2])
        ys2 = m * y[:,0] + c
        #if np.min(y[:,0] <= np.max(xs)) and (np.max(y[:,0] >= np.min(xs))):
        plt.plot(y[:,0],y[:,1])
	#plt.plot(y[:,0],ys)
        idx = np.argwhere(np.isclose(y[:,1], ys2, atol=1)).reshape(-1)
        if np.size(idx) > 1:
            idx = idx[0]
        if (y[idx,0]  <= np.max(xs)) and (y[idx,0] >= np.min(xs)):
	#this works out the intersection with each line...
            plt.plot(y[idx,0], ys2[idx], 'ro')
            locations_x.append(y[idx,0])
            locations_y.append(ys2[idx])


plt.scatter(x0,y0)
plt.scatter(x1,y1)
plt.plot(xs,ys)
plt.xlim([0,image_size[1]])
plt.ylim([0,image_size[0]])
#plt.show(block = False)

distances = np.zeros(np.size(locations_x)-1)
for i in range(np.size(locations_x)-1):
    x1 = locations_x[i]
    x2 = locations_x[i+1]
    y1 = locations_y[i]
    y2 = locations_y[i+1]
    distances[i] = sqrt( (x2 - x1)**2 + (y2 - y1)**2 )

plt.subplot(2, 1, 2)
plt.plot(distances)
plt.xlabel('growth band number')
plt.ylabel('distance')
plt.show(block = False)



'''
Note, We need to somehow specify that we are moving out from a certain point, then calculat eteh shorttest distance between each point on that line and the next line, then take the longest shortest point (Or work out how to do ditances normal to the tangent for each point?)
'''
