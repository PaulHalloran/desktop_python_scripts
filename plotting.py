import numpy as np
import matplotlib.pyplot as plt
import iris

#file='/net/home/h04/hador/python_scripts/PythonScientists/data/yeoviltondata.txt'

#data=np.genfromtxt(file,skip_header=6)

#ax = plt.figure().add_subplot(211)
#ax.plot(data[:,0]+data[:,1]/12.0,data[:,2],label='what is this')
#leg = ax.legend(fancybox=True,bbox_to_anchor=(1.0, 1.2))
#leg.draw_frame(False)
#ax.grid(True)
#plt.savefig('/data/local/hador/delete.ps')
#plt.show()

#BARS=50
#ax = plt.figure().add_subplot(111)
#ax.hist(normal,BARS,color='red')
#ax.set_ylim(0,SAMPLE*5.0/BARS)
#ax.set_xlim(-5,5)
#ax.set_xlabel('x values')
#ax.set_ylabel('y values')
#plt.savefig('/data/local/hador/delete.png')
#plt.show()


filename = iris.sample_data_path('hybrid_height.pp')
cube = iris.load_cube(filename)
print cube
equator_slice = cube.extract(iris.Constraint(grid_latitude=0))
print equator_slice
