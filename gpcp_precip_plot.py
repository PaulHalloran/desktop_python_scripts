import iris
import matplotlib.pyplot as plt
import iris.quickplot as qplt
import numpy as np

x = iris.load_cube('/home/ph290/data1/observations/precip/precip.mon.mean.nc')
x2 = x.collapsed('time',iris.analysis.MEAN)

plt.figure(num=None, figsize=(8, 6), dpi=150, facecolor='w', edgecolor='k')
qplt.contourf(x2,np.linspace(0,10,50))
plt.gca().coastlines()
plt.show()

