import matplotlib.pyplot as plt
import numpy as np
from matplotlib.mlab import griddata
import cartopy.crs as ccrs
import scipy
import scipy.stats
import cartopy



file = '/data/data1/ph290/observations/dms_i0e832.dat'

data = np.genfromtxt(file,skip_header = 10,delimiter = '\t', skip_footer = 50900-50500-10)

lats = np.floor(data[:,2])
lons = np.floor(data[:,3])
swdms = data[:,4]
swdms[np.where(swdms == -999.0)] = np.NAN

final_dms = np.zeros([180,360])
final_dms[:] = np.NAN
dms_no_obs = final_dms.copy()

for i in range(-180,180):
	print i
	for j in range(-90,90):
		loc = np.where((lats == j) & (lons == i))
		if np.size(loc) > 0:
			#print loc
			#print np.mean(swdms[loc])
			final_dms[j,i] = scipy.stats.nanmean(swdms[loc])
			dms_no_obs[j,i] = np.size(loc)


lons2, lats2 = np.meshgrid(range(-180,180), range(-90,90))


final_dms = np.ma.masked_invalid(final_dms)
dms_no_obs = np.ma.masked_invalid(dms_no_obs)
max_val = 20
ax = plt.axes(projection=ccrs.PlateCarree())
CS = ax.pcolormesh(lons2,lats2,np.roll(np.roll(final_dms,90,axis=0),180,axis=1),vmin=0,vmax=max_val,transform=ccrs.PlateCarree())
ax.coastlines()
ax.set_global()
ax.add_feature(cartopy.feature.LAND)
cb1 = plt.colorbar(CS,orientation = 'horizontal')  # draw colorbar
cb1.set_label('nM DMS or number of observations')
plt.savefig('/home/ph290/Documents/figures/dms1.pdf')

ax = plt.axes(projection=ccrs.PlateCarree())
CS = ax.pcolormesh(lons2,lats2,np.roll(np.roll(dms_no_obs,90,axis=0),180,axis=1),vmin=0,vmax=max_val,transform=ccrs.PlateCarree())
ax.coastlines()
ax.set_global()
ax.add_feature(cartopy.feature.LAND)
#plt.colorbar(CS,orientation = 'horizontal')  # draw colorbar
ax.set_extent([-90, -50, -80, -45])
plt.savefig('/home/ph290/Documents/figures/dms2.pdf')