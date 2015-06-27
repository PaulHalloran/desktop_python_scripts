import matplotlib.pyplot as plt
import carbchem
import numpy as np
import random
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy.ma as ma
from numpy.random import uniform, seed


mdi=-999.0
sizing=(2)
T = np.empty(sizing)
S = np.empty(sizing)
TCO2 = np.empty(sizing)
TALK = np.empty(sizing)
T.fill(15.0)
S.fill(32.0)

size = 10000

tco2_store = np.empty(size)
talk_store = np.empty(size)
ph_store = np.empty(size)
co3_store = np.empty(size)
co2_store = np.empty(size)

for i in range(size):
	tmp_tco2 = random.randint(1600,2300)/1.0e6
	tmp_alk = random.randint(1800,2500)/1.0e6
	tco2_store[i] = tmp_tco2
	talk_store[i] = tmp_alk
	TCO2.fill(tmp_tco2)
	TALK.fill(tmp_alk)
	ph_store[i] =  carbchem.carbchem(2,mdi,T,S,TCO2,TALK)[0]
	co3_store[i] =  carbchem.carbchem(5,mdi,T,S,TCO2,TALK)[0]
	co2_store[i] = carbchem.carbchem(1,mdi,T,S,TCO2,TALK)[0]



x = ph_store
y = co3_store
z = co2_store
# define grid.
xi = np.linspace(np.min(x),np.max(x),100)
yi = np.linspace(np.min(y),np.max(y),100)
# grid the data.
zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
# contour the gridded data, plotting dots at the randomly spaced data points.
#CS = plt.contour(xi,yi,zi,51,linewidths=0.5,colors='k')
CS = plt.contourf(xi,yi,zi,np.linspace(100,3000,51),cmap=plt.cm.jet)
plt.colorbar() # draw colorbar
# plot data points.
#plt.scatter(x,y,marker='o',c='b',s=5)
#plt.ylim(-2,2)
#plt.title('griddata test (%d points)' % npts)
plt.show()




