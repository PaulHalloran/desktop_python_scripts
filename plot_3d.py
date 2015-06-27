from numpy import *
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

tmp=arange(10)
#for j in tmp[::-1]:
no=100.0
for i in arange(no):
	if (i == 0) | ((i > 50) & (i < 96)):
		# Set legend font size (optional):
		mpl.rcParams['legend.fontsize'] = 15
		# Setup the 3D plot:
		fig = plt.figure()
		ax = fig.gca(projection='3d')
		# Define interval for parameter 't' and its # division:
		t = linspace(0, 1, 10)
		# Define the curve:
		x = (1 + t**2) * sin(2 * pi * t)
		y = (1 + t**2) * cos(2 * pi * t)

		print i
		x2 = (((1 + t**2) * sin(2 * pi * t))/(no-(i+1.0)))-mean(((1 + t**2) * sin(2 * pi * t))/(no-(i+1.0)))
		y2 = (((1 + t**2) * cos(2 * pi * t))/(no-(i+1.0)))-mean(((1 + t**2) * cos(2 * pi * t))/(no-(i+1.0)))
	
		z=t
		x1=zeros(size(t))
		y1=zeros(size(t))
		z1=z
		# Plot the curve and show the plot:
		ax.plot(x1, y1, z1, label='Parametric 3D curve')
		#ax.scatter(x, y, z, label='Parametric 3D curve')
		#ax.scatter(x2, y2, z,label='Parametric 3D curve')
		for k in arange(10)/5.0:
			for k2 in arange(10)/5.0:
				ax.scatter(-1+x2+k, -1+y2+k2, z,label='Parametric 3D curve')
		ax.set_ylim3d(-2,2)
		ax.set_xlim3d(-2,2)
		#ax.legend()
		#plt.show()
		plt.savefig('/Users/ph290/Documents/animation/test_0_'+str(i/10.0)+'.png')

no=100.0
for l in arange(50):
	i=96
	print l
	# Set legend font size (optional):
	mpl.rcParams['legend.fontsize'] = 15
	# Setup the 3D plot:
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	# Define interval for parameter 't' and its # division:
	t = linspace(0, 1, 10)
	# Define the curve:
	x = (1 + t**2) * sin(2 * pi * t)
	y = (1 + t**2) * cos(2 * pi * t)
	x2 = (((1 + t**2) * sin(2 * pi * t))/(no-(i+1.0)))-mean(((1 + t**2) * sin(2 * pi * t))/(no-(i+1.0)))
	y2 = (((1 + t**2) * cos(2 * pi * t))/(no-(i+1.0)))-mean(((1 + t**2) * cos(2 * pi * t))/(no-(i+1.0)))
	z=t
	x1=zeros(size(t))
	y1=zeros(size(t))
	z1=z
	# Plot the curve and show the plot:
	ax.plot(x1, y1, z1, label='Parametric 3D curve')
	#ax.scatter(x, y, z, label='Parametric 3D curve')
	#ax.scatter(x2, y2, z,label='Parametric 3D curve')
	for k in arange(10)/5.0:
		for k2 in arange(10)/5.0:
			ax.scatter(-1+x2+k, -1+y2+k2, z,label='Parametric 3D curve')
	ax.set_ylim3d(-2,2)
	ax.set_xlim3d(-2,2)
	#ax.legend()
	#plt.show()
	plt.savefig('/Users/ph290/Documents/animation/test_part_b_0_'+str(l/10.0)+'.png')	
	
	