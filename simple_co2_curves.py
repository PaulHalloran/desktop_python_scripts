'''
4th?? order polynomial to rcp 85 - play with to come up with my idealised scenarios
'''
import numpy as np
import matplotlib.pyplot as plt

'''

directory = '/home/ph290/box_modelling/boxmodel_6_box_back_to_basics/forcing_data/co2/'
data = np.genfromtxt(directory+'historical_and_rcp85_atm_co2.txt',delimiter=',')
loc = np.where((1860 <= data[:,0]) & (data[:,0] <= 2100))
data = data[loc,:]
data=data[0]
z = np.polyfit(data[:,0], data[:,1], 3)
polynomial = np.poly1d(z)
ys = polynomial(data[:,0])

#offset = [0.99995,0.9999]
#offset = [1.0]
offset = [0.8,1,1.2]

plt.close('all')
plt.figure()

out_data = np.zeros([np.size(data[:,0]),2])

for i,off in enumerate(offset):

#     z = np.polyfit(data[:,0], data[:,1], 4)
#     polynomial = np.poly1d(z)
#     ys = polynomial(data[:,0])

#     poly2 = polynomial
# #    poly2[2] /= off
# #    poly2[1] /= off
#     poly2[1] /= 1.00
#     poly2[2] /= 1.000
#     poly2[3] /= 1.00
#     poly2[4] /= off
#     ys2 = poly2(data[:,0])

#     plt.plot(data[:,0],data[:,1])
#     plt.plot(data[:,0],ys)
#     plt.plot(data[:,0],ys2-ys2[0]+ys[0])
#     plt.show(block = False)



    #out_data[:,0] = data[:,0]
    #out_data[:,1] = data[:,1]


    #np.savetxt(directory+'rcp85_1.txt', out_data, fmt=['%4.1f','%11.8f'],delimiter=',')  
    out_data[:,0] = data[:,0]
    tmp = data[:,1]*off
    out_data[:,1] = tmp-tmp[0]+data[0,1]
    np.savetxt(directory+'rcp85_'+np.str(i+1)+'.txt', out_data, fmt=['%4.1f','%11.8f'],delimiter=',')  

    plt.plot(out_data[:,0],out_data[:,1])

plt.show(block = False)]

'''

x = data[:,0]-data[0,0]
y = data[:,1]

plt.close('all')

plt.figure()
plt.plot(x,y,'b')
plt.plot(x,y[0]+1.0285**x,'r')
plt.plot(x,y[0]+1.0265**x,'r--')
plt.plot(x,y[0]+1.0305**x,'r-.')
plt.show(block = False)

out_data = np.zeros([np.size(data[:,0]),2])
out_data[:,0] = data[:,0]
out_data[:,1] = y[0]+1.0285**x
np.savetxt(directory+'rcp85_'+np.str(1)+'.txt', out_data, fmt=['%4.1f','%11.8f'],delimiter=',')  
out_data[:,1] = y[0]+1.0265**x
np.savetxt(directory+'rcp85_'+np.str(2)+'.txt', out_data, fmt=['%4.1f','%11.8f'],delimiter=',')
out_data[:,1] = y[0]+1.0305**x
np.savetxt(directory+'rcp85_'+np.str(3)+'.txt', out_data, fmt=['%4.1f','%11.8f'],delimiter=',')  
