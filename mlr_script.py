import numpy as np
from numpy.random import random

n=50
k=20
'''
y = np.mat(random((1,n)))
'''
X = np.mat(random((k,n)))

b = y * X.T * np.linalg.inv(X*X.T)
print(b)

a = np.array(X)[0]*0.0
for i in np.arange(k):
	a2 = np.array(X)[i]*np.array(b)[0][i]
	a += a2

plt.figure(num=None, figsize=(10, 3), dpi=300, facecolor='w', edgecolor='k')
plt.plot(np.array(y)[0])
plt.plot(a)
plt.xlabel('variable x')
plt.ylabel('variable y')
plt.tight_layout()
plt.savefig('/Users/ph290/Documents/teaching/GEO3228AnnesModellingModule/figures/mlr_'+np.str(k)+'.png')

plt.figure(num=None, figsize=(10, 4), dpi=300, facecolor='w', edgecolor='k')
for i in np.arange(k):
	plt.plot(np.array(x)[i])
plt.xlabel('variable x')
plt.ylabel('variable y')
plt.tight_layout()
plt.savefig('/Users/ph290/Documents/teaching/GEO3228AnnesModellingModule/figures/mlr_input_'+np.str(k)+'.png')


