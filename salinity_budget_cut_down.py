'''
USE Sally E. Close and Hugues Goosse JOURNAL OF GEOPHYSICAL RESEARCH: OCEANS, VOL 118 2811-2827 2013
'''

from iris.coords import DimCoord
import iris.plot as iplt
import time
import numpy as np
import glob
import iris
import iris.coord_categorisation
import iris.analysis
import subprocess
import os
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import numpy.ma as ma
import running_mean as rm
import running_mean_post as rmp
from scipy import signal
import scipy
import scipy.stats
import numpy as np
import statsmodels.api as sm
import running_mean_post
from scipy.interpolate import interp1d
from datetime import datetime, timedelta
import cartopy.crs as ccrs
import iris.analysis.cartography
import numpy.ma as ma
import scipy.interpolate
import gc
import pickle
import biggus
import seawater
import cartopy.feature as cfeature
import statsmodels.api as sm
from eofs.iris import Eof
import cartopy.feature as cfeature
import cartopy
import cartopy.crs as ccrs
import matplotlib as mpl
from scipy.stats import gaussian_kde
from statsmodels.stats.outliers_influence import summary_table
import scipy.ndimage
import scipy.ndimage.filters
import gsw
import scipy.stats as stats



###
#Filter
###


def butter_bandpass(lowcut, fs, order=5):
    nyq = fs
    low = lowcut/nyq
    b, a = scipy.signal.butter(order, low , btype='high',analog = False)
    return b, a


def butter_lowpass(lowcut, fs, order=5):
    nyq = fs
    low = lowcut/nyq
    b, a = scipy.signal.butter(order, low , btype='high',analog = False)
    return b, a



def butter_highpass(highcut, fs, order=5):
    nyq = fs
    high = highcut/nyq
    b, a = scipy.signal.butter(order, high , btype='low',analog = False)
    return b, a




def mld(S,thetao,depth_cube,latitude_deg):
	"""Compute the mixed layer depth.
	Parameters
	----------
	SA : array_like
		 Absolute Salinity  [g/kg]
	CT : array_like
		 Conservative Temperature [:math:`^\circ` C (ITS-90)]
	p : array_like
		sea pressure [dbar]
	criterion : str, optional
			   MLD Criteria
	Mixed layer depth criteria are:
	'temperature' : Computed based on constant temperature difference
	criterion, CT(0) - T[mld] = 0.5 degree C.
	'density' : computed based on the constant potential density difference
	criterion, pd[0] - pd[mld] = 0.125 in sigma units.
	`pdvar` : computed based on variable potential density criterion
	pd[0] - pd[mld] = var(T[0], S[0]), where var is a variable potential
	density difference which corresponds to constant temperature difference of
	0.5 degree C.
	Returns
	-------
	MLD : array_like
		  Mixed layer depth
	idx_mld : bool array
			  Boolean array in the shape of p with MLD index.
	Examples
	--------
	>>> import os
	>>> import gsw
	>>> import matplotlib.pyplot as plt
	>>> from oceans import mld
	>>> from gsw.utilities import Bunch
	>>> # Read data file with check value profiles
	>>> datadir = os.path.join(os.path.dirname(gsw.utilities.__file__), 'data')
	>>> cv = Bunch(np.load(os.path.join(datadir, 'gsw_cv_v3_0.npz')))
	>>> SA, CT, p = (cv.SA_chck_cast[:, 0], cv.CT_chck_cast[:, 0],
	...              cv.p_chck_cast[:, 0])
	>>> fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, sharey=True)
	>>> l0 = ax0.plot(CT, -p, 'b.-')
	>>> MDL, idx = mld(SA, CT, p, criterion='temperature')
	>>> l1 = ax0.plot(CT[idx], -p[idx], 'ro')
	>>> l2 = ax1.plot(CT, -p, 'b.-')
	>>> MDL, idx = mld(SA, CT, p, criterion='density')
	>>> l3 = ax1.plot(CT[idx], -p[idx], 'ro')
	>>> l4 = ax2.plot(CT, -p, 'b.-')
	>>> MDL, idx = mld(SA, CT, p, criterion='pdvar')
	>>> l5 = ax2.plot(CT[idx], -p[idx], 'ro')
	>>> _ = ax2.set_ylim(-500, 0)
	References
	----------
	.. [1] Monterey, G., and S. Levitus, 1997: Seasonal variability of mixed
	layer depth for the World Ocean. NOAA Atlas, NESDIS 14, 100 pp.
	Washington, D.C.
	"""
	#depth_cube.data = np.ma.masked_array(np.swapaxes(np.tile(depths,[360,180,1]),0,2))
	try:
		S.coord('depth')
		MLD_out = S.extract(iris.Constraint(depth = np.min(depth_cube.data)))
	except:
		MLD_out = S[:,0,:,:]
	MLD_out_data = MLD_out.data
	for i in range(np.shape(MLD_out)[0]):
		print'calculating mixed layer for year: ',i
		thetao_tmp = thetao[i]
		S_tmp = S[i]
		depth_cube.data = np.abs(depth_cube.data)
		depth_cube = depth_cube * (-1.0)
		p = gsw.p_from_z(depth_cube.data,latitude_deg.data) # dbar
		SA = S_tmp.data*1.004715
		CT = gsw.CT_from_pt(SA,thetao_tmp.data - 273.15)
		SA, CT, p = map(np.asanyarray, (SA, CT, p))
		SA, CT, p = np.broadcast_arrays(SA, CT, p)
		SA, CT, p = map(ma.masked_invalid, (SA, CT, p))
		p_min, idx = p.min(axis = 0), p.argmin(axis = 0)
		sigma = SA.copy()
		to_mask = np.where(sigma == S.data.fill_value)
		sigma = gsw.rho(SA, CT, p_min) - 1000.
		sigma[to_mask] = np.NAN
		sig_diff = sigma[0,:,:].copy()
		sig_diff += 0.125 # Levitus (1982) density criteria
		sig_diff = np.tile(sig_diff,[np.shape(sigma)[0],1,1])
		idx_mld = sigma <= sig_diff
		#NEED TO SORT THS PIT - COMPARE WWITH OTHER AND FIX!!!!!!!!!!
		MLD = ma.masked_all_like(S_tmp.data)
		MLD[idx_mld] = depth_cube.data[idx_mld] * -1
		MLD_out_data[i,:,:] = np.ma.max(MLD,axis=0)
	return MLD_out_data



models = ['MIROC-ESM','MRI-CGCM3','MPI-ESM-P', 'GISS-E2-R','CSIRO-Mk3L-1-2', 'HadCM3']
#models = ['CCSM4']
print 'I NEED TO DOWNLOAD MORE CCSM4 SO FILES ONCE ESGF BACK UP:'

print 'so_Omon_CCSM4_past1000_r1i1p1_144001-144912.nc'
print 'so_Omon_CCSM4_past1000_r1i1p1_145001-145912.nc'
print 'so_Omon_CCSM4_past1000_r1i1p1_146001-146912.nc'

#NOTE MIROC currently crashes things (looks like a memory issue - segmentation sault) - find out why
#models = ['MPI-ESM-P', 'GISS-E2-R','CSIRO-Mk3L-1-2', 'HadCM3']

directory = '/data/NAS-ph290/ph290/cmip5/last1000/'


######################################
#   Calculating Mixed layer depths   #
######################################

ensembles = ['r1i1p1','r1i1p121']

print 'Calculating mixed layer depths'


for ensemble in ensembles:
	for model in models:
		print model
		try:
			test = glob.glob(directory+model+'_my_mld_'+ensemble+'.nc')
			if np.size(test) == 0:
				print 'calculating MLD for '+model
				S = iris.load_cube(directory+model+'*_so_past1000_'+ensemble+'_*Omon*.nc')
				thetao = iris.load_cube(directory+model+'*_thetao_past1000_'+ensemble+'_*Omon*.nc')
				#S = S[0:2]
				#thetao = thetao[0:2]
				depth_cube = S[0].copy()
				try:
					S.coord('depth')
					depths = depth_cube.coord('depth').points
				except:
					depths = depth_cube.coord('ocean sigma over z coordinate').points
				 #for memorys sake, do one year at a time..
				depth_cube.data = np.ma.masked_array(np.swapaxes(np.tile(depths,[360,180,1]),0,2))
				try:
					S.coord('depth')
					hm = S.extract(iris.Constraint(depth = np.min(depth_cube.data)))
				except:
					hm = S[:,0,:,:]
				latitude_deg = depth_cube.copy()
				latitude_deg_data = hm.coord('latitude').points
				latitude_deg.data = np.swapaxes(np.tile(latitude_deg_data,[np.shape(S)[1],360,1]),1,2)
				hm.data = mld(S,thetao,depth_cube,latitude_deg)
				iris.fileformats.netcdf.save(hm,directory+model+'_my_mld_'+ensemble+'.nc')
			else:
				print 'MLD for '+model+' already exists'
		except:
			print ensemble+' '+model+' failed'


model_data = {}
ensemble = 'r1i1p1'

##############################
#     constants              #
##############################

#number of seconds in a year
yearsec = 60.0 * 60.0 * 24.0 * 360.0
# eddy diffusivity - held constant, following Dong et al. [2009],
k = 500 #m2/s
#  characteristic density of the mixed layer (taken here to be 1027kg m3)
po = 1027.0
# density of sea ice, estimated to be 930 kg m-3
pi = 930.0
# salinity of sea ice, estimated here as 5
Si = 5
#note that upside delta (nabia) is the gradient and can be calculated with np.gradient() but this is calculates in all three dimentsin (time, x and y), so to get what you waht use:
#tmp = np.gradient(variable_of_interest.data)
#gradient_of_variable.data = tmp[1]+tmp[2]
# nabia squared is called the laplacian and can be calculated with scipy.ndimage.filters.laplace() I think

##############################
#     salinity budget        #
##############################

print 'Calculating salinity budget'

###########################################
#    restrict loading to top 1000m        #
###########################################

chosen_levels = lambda cell: cell <= 1000
level_above_1000 = iris.Constraint(depth=chosen_levels)



for model in models:
    print model
    model_data[model] = {}
    ######################################
    # Sea ice concentration              #
    ######################################
    sic = iris.load_cube(directory+model+'*_sic_past1000_r1i1p1_*.nc')
    ######################################
    # P minus E                          #
    ######################################
    try:
        # mass of water vapor evaporating from the ice-free portion of the ocean
        E_no_ice = iris.load_cube(directory+model+'*_evs_past1000_r1i1p1_*Omon*.nc')
        # mass of liquid water falling as liquid rain  into the ice-free portion of the ocean
        P_no_ice_rain = iris.load_cube(directory+model+'*_pr_past1000_r1i1p1_*Omon*.nc')
        # mass of solid water falling as liquid rain  into the ice-free portion of the ocean
        P_no_ice_snow = iris.load_cube(directory+model+'*_prsn_past1000_r1i1p1_*Omon*.nc')
        P_no_ice = P_no_ice_rain + P_no_ice_snow
        P_minus_E_no_ice = P_no_ice - E_no_ice
        P_minus_E_no_ice *= yearsec
        P = P_no_ice * yearsec
        E = E_no_ice * yearsec
        #convert into a flux per year
    except:
        # mass of water vapor evaporating from all portion of the ocean
        E = iris.load_cube(directory+model+'*_evspsbl_past1000_r1i1p1_*.nc')
        # mass of solid+liquid water falling into the whole of the ocean
        P = iris.load_cube(directory+model+'*_pr_past1000_r1i1p1_*Amon*.nc')
        P_minus_E_no_ice = (P - E) * (((sic*-1.0)+100.0)/100.0)
        #convert into a flux per year
        P_minus_E_no_ice *= yearsec
        P = (P * (((sic*-1.0)+100.0)/100.0)) * yearsec
        E = (E * (((sic*-1.0)+100.0)/100.0)) * yearsec
        evap_precip_flag = True
	######################################
	# 3D salinity                        #
	######################################
	S = iris.load_cube(directory+model+'*_so_past1000_r1i1p1_*Omon*.nc')
	S.coord(dimensions=1).rename('depth')
	S = S.extract(level_above_1000)
	#units = psu = g/kg = kg/1000kg. Convert it to kg m-3
	S /= 1.026 # Change this so it uses the actual grid point density
	######################################
	# hight of the mixed layer (mld)     #
	######################################
	hm = iris.load_cube(directory+model+'_my_mld_'+ensemble+'.nc')
	######################################
	# mixed layer salinity               #
	######################################
	depth_cube = S[0].copy()
	depths = depth_cube.coord('depth').points
	depth_cube.data = np.ma.masked_array(np.swapaxes(np.tile(depths,[360,180,1]),0,2))
	thickness_cube = S[0].copy()
	thicknesses = depth_cube.coord('depth').bounds[:,1] - depth_cube.coord('depth').bounds[:,0]
	thickness_cube.data = np.ma.masked_array(np.swapaxes(np.tile(thicknesses,[360,180,1]),0,2))
	#  mask everywhere below the mixed layer and calculate the mean mixed layer salinity #
	s_mixed_layer = S.extract(iris.Constraint(depth = depths[0]))
	s_mixed_layer_data = s_mixed_layer.data.copy()
	for time_index,cube_tmp in enumerate(S.slices(['depth','latitude','longitude'])):
		print time_index
		thickness_cube2 = thickness_cube.copy()
		thickness_cube2_data = thickness_cube2.data
		tmp_data = cube_tmp.data
		for depth in np.arange(np.size(depths)):
			tmp_data[depth,:,:] = np.ma.masked_where(hm[time_index,:,:].data < depths[depth],tmp_data[depth,:,:])
			thickness_cube2_data[depth,:,:] = np.ma.masked_where(hm[time_index,:,:].data < depths[depth],thickness_cube2_data[depth,:,:])
		cube_tmp.data = tmp_data
		cube_tmp *= thickness_cube
		s_mixed_layer_data[time_index,:,:] = cube_tmp.collapsed(['depth'],iris.analysis.SUM).data
		thickness_cube2.data = thickness_cube2_data
		total_thickness_cube = thickness_cube2.collapsed(['depth'],iris.analysis.SUM).data
		s_mixed_layer_data[time_index,:,:] /= total_thickness_cube.data
	#####################################################################
	# s_mixed_layer contains the average salinity of the mixed layer    #
	#####################################################################
	s_mixed_layer.data = s_mixed_layer_data
	######################################
	######################################
	### Components of salinity change  ###
	######################################
	######################################
	#E-P driven salinity change (1st, kg freshwater water added per m-3 per year)
	sw_density = 1.026 # (but should really change to use local ML averaged density)
	EP_contribution = (P_minus_E_no_ice / hm)
	#Fraction reduction in salinity
	EP_contribution = EP_contribution / (1000.0 * sw_density)
	#PSU reduction in salinity associated with P-E
	EP_contribution = (-1.0) * (EP_contribution * s_mixed_layer)
    P = (-1.0) * (((P / hm) / (1000.0 * sw_density)) * s_mixed_layer)
    E = (-1.0) * (((E / hm) / (1000.0 * sw_density)) * s_mixed_layer)
    model_data[model]['mixed_layer_salinity'] = s_mixed_layer
	#NOTE RUNS DSO FAR HAVE BEEN 1e3 too low - had an spurios divide by 1000. Now removed for any future anlaysusl/...
    model_data[model]['EP_contribution'] = EP_contribution
    model_data[model]['P'] = P
    model_data[model]['E'] = E
    model_data2 = model_data
    with open('/data/NAS-ph290/ph290/cmip5/pickles/salinity_budget_plot_6.pickle', 'w') as f:
        pickle.dump([models,model_data2], f)


#with open('/data/NAS-ph290/ph290/cmip5/pickles/salinity_budget_plot_2.pickle', 'w') as f:
#   pickle.dump([models,model_data], f)


# with open('/data/NAS-ph290/ph290/cmip5/pickles/salinity_budget_plot_4.pickle', 'r') as f:
#    models,model_data = pickle.load(f)


'''

def extract_and_mean(cube):
	west = -24
	east = -13
	south = 65
	north = 67
	cube = cube.intersection(longitude=(west, east))
	cube = cube.intersection(latitude=(south, north))
	try:
		cube.coord('latitude').guess_bounds()
	except:
		print 'cube already has latitude bounds'
	try:
		cube.coord('longitude').guess_bounds()
	except:
		print 'cube already has longitude bounds'
	grid_areas = iris.analysis.cartography.area_weights(cube)
	return cube.collapsed(['latitude','longitude'],iris.analysis.MEAN, weights=grid_areas)


start_date = 850
end_date = 1850
tmp_years = np.arange(start_date,end_date+1)


b1, a1 = butter_lowpass(1.0/100.0, 1.0,2)
b2, a2 = butter_highpass(1.0/10, 1.0,2)


def average_accross_models(tmp_years,model_data,models,variable):
	out = np.empty([np.size(models),np.size(tmp_years)]) * 0.0 + np.NAN
	for i,model in enumerate(models):
		data = extract_and_mean(model_data[model][variable]).data
		coord = model_data[model][variable].coord('time')
		dt = coord.units.num2date(coord.points)
		year = np.array([coord.units.num2date(value).year for value in coord.points])
		data = scipy.signal.filtfilt(b1, a1, data)
		data = scipy.signal.filtfilt(b2, a2, data)
		data = (data-np.min(data))/(np.max(data)-np.min(data))
		for j,yr in enumerate(tmp_years):
			loc = np.where(year == yr)
			if np.size(loc) != 0:
				out[i,j] = data[loc]
	return scipy.stats.nanmean(out,axis = 0)


mixed_layer_salinity_mean = average_accross_models(tmp_years,model_data,models,'mixed_layer_salinity')
EP_contribution_mean = average_accross_models(tmp_years,model_data,models,'EP_contribution')



plt.close('all')
plt.plot(tmp_years,mixed_layer_salinity_mean - np.mean(mixed_layer_salinity_mean),'b',label = 'multi model mean mixed layer salinity')
plt.plot(tmp_years,EP_contribution_mean - np.mean(EP_contribution_mean),'r',label = 'multi model mean P-E driven salinity change')
plt.xlabel('year')
plt.ylabel('kg salt m$^{-3}$')
plt.legend()
plt.savefig('/home/ph290/Documents/figures/sal_and_pe.png')
#plt.show(block = False)


plt.close('all')
x = mixed_layer_salinity_mean - np.mean(mixed_layer_salinity_mean)
y = EP_contribution_mean - np.mean(EP_contribution_mean)
plt.scatter(x,y)
#plt.savefig('/home/ph290/Documents/figures/sal_and_pe_scatter.png')
plt.show(block = False)


'''
