import matplotlib.pyplot as plt

import cartopy.crs as ccrs

ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))

# make the map global rather than have it zoom in to
# the extents of any plotted data
ax.set_global()

ax.stock_img()
ax.coastlines()

#plt.show()
plt.savefig('/home/ph290/Documents/figures/world_PlateCarree.png')



ax = plt.axes(projection=ccrs.LambertCylindrical(central_longitude=180))

# make the map global rather than have it zoom in to
# the extents of any plotted data
ax.set_global()

ax.stock_img()
ax.coastlines()

plt.savefig('/home/ph290/Documents/figures/world_lamb.png')
