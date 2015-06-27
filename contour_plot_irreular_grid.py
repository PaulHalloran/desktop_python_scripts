import cartopy.crs as ccrs
import iris
import iris.quickplot as qplt
import matplotlib.pyplot as plt

def contour_plot_irreular_grid(cube):
    var_regridded, extent = iris.analysis.cartography.project(cube, ccrs.PlateCarree())
    qplt.contourf(var_regridded)
    plt.show()




