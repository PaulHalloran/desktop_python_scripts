import numpy as np

def running_mean(x, N):
	y = np.zeros((len(x),))
	for ctr in range(len(x)):
		y[ctr] = np.sum(x[(ctr-np.round(N/2)):ctr+np.round(N/2)])
	out = y/N
	out[0:np.round(N/2)] = np.NAN
	out[-1.0*np.round(N/2)::] = np.NAN
	return out


