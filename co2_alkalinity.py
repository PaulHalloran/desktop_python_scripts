import carbchem
import numpy as np
import matplotlib.pyplot as plt

mdi=-999.0
sizing=(100)
T = np.empty(sizing)
S = np.empty(sizing)
TCO2 = np.empty(sizing)
TALK = np.empty(sizing)
T.fill(15.0)
S.fill(32.0)

tco2_tmp = 0.005
talk_tmp = 0.005

for i in np.arange(100):
	TCO2[i] = tco2_tmp
	TALK[i] = talk_tmp
	tco2_tmp = tco2_tmp*0.99
	talk_tmp = talk_tmp*0.99

plt.plot(TALK,carbchem.carbchem(1,mdi,T,S,TCO2,TALK))
plt.show()