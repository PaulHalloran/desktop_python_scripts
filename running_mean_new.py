import numpy as np

def running_mean_post(x, N):
	y = np.zeros((len(x),))
	for ctr in range(len(x)):
		y[ctr] = np.sum(x[(ctr-np.round(N/2)):ctr+np.round(N/2)])
	return y/N


