import iris
import iris.quickplot as qplt
import iris.plot as iplt
import matplotlib.pyplot as plt
import numpy as np
import re
import string
import itertools

input = '[AAAA]'
n1 = len(input.split(' ')[0]) - 2
s1 = input[n1+2:]
names = [''.join(x)+s1 for x in itertools.product(string.ascii_lowercase, repeat=n1)]

#data from http://cdiac.ornl.gov/ftp/oceans/GLODAP_Gridded_Data/
cube = iris.load('/home/ph290/Downloads/CFC11/CFC11.nc','CFC-11')
cube = cube[0]
cube.coord('depth').points=cube.coord('depth').points*(-1.0)

for i in np.arange(359):
     plt.figure()
     meridional_slice2 = cube.extract(iris.Constraint(longitude=-179.5+i))
     qplt.contourf(meridional_slice2, levels = np.linspace(0.0,8,50), coords=['latitude','depth'])
     #plt.show()
     plt.savefig('/home/ph290/Downloads/anim3/anim_'+names[i]+'.png')

for i in np.arange(359):
    plt.figure()
    iplt.contourf(cube[0], 50)
    plt.gca().coastlines()
    plt.plot([-180+i,-180+i],[-90,90],'k', linewidth = 5)
    plt.savefig('/home/ph290/Downloads/anim4/anim_'+names[i]+'.png')
