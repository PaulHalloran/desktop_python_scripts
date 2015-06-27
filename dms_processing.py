import numpy as np
import matplotlib.pyplot as plt
import iris
import iris.analysis
import iris.analysis.cartography


filenames='/data/local/hador/mass_retrievals/dms/*/*.pp'
cube = iris.load(filenames)
cube=cube[0]

cube.coord('latitude').guess_bounds()
cube.coord('longitude').guess_bounds()
grid_areas = iris.analysis.cartography.area_weights(cube)

mean = cube.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas)

plt.plot(mean.data)
plt.show()
