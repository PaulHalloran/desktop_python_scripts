import carbchem
import matplotlib.pyplot as plt
import numpy as np

sizing = 1000
mdi=-999.0
T = np.empty(sizing)
S = np.empty(sizing)
TCO2 = np.empty(sizing)
TALK = np.empty(sizing)
T.fill(15.0)
S.fill(32.0)
TCO2 = np.linspace(0.0016,0.0023,1000)
TALK.fill(0.0024)

co2 = carbchem.carbchem(1,mdi,T,S,TCO2,TALK)

plt.plot(co2,TCO2,'r')

TALK.fill(0.0020)

co2 = carbchem.carbchem(1,mdi,T,S,TCO2,TALK)

plt.plot(co2,TCO2,'b')
plt.xlim([200,1000])
#plt.show(block = True)
plt.savefig('/home/ph290/Documents/figures/co2_dic.pdf')




