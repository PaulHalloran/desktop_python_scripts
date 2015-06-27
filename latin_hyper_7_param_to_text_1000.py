import numpy as np

in_file='/data/data1/ph290/code/r_scripts/lhs.txt'
input=np.genfromtxt(in_file)

input_shape=input.shape

no_param_sets=input_shape[1]

#;--------------parameter set------------;

kgas_sp=(input[:,0]*0.4)+0.00001
kgas_s=(input[:,1]*0.4)+0.00001
kgas_tr=(input[:,2]*0.4)+0.00001
mixing=(input[:,3]*20.0)+0.00001
mixing2=(input[:,4]*20.0)+0.00001
alpha_in=input[:,5]*input[:,6]
beta_in=(1.0-input[:,5])*input[:,6]


np.savetxt('/home/ph290/data1/data/piston_hypercube_7param_1000.csv',np.vstack((kgas_sp,kgas_s,kgas_tr,mixing,mixing2,alpha_in,beta_in)).T, delimiter=',')

