import numpy as np
import glob
import iris
import numpy.ma as ma
import gsw


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
	MLD_out = S.extract(iris.Constraint(depth = np.min(depth_cube.data)))
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



def cmip5_mixed_layers(S,thetao):
	##############################
	# 	inputs:
	# 	S = CMIP5 3D+time salinity cube (so)
	# 	theta = CMIP5 3D+time temperature cube (thetao)
	# 	returns:
	# 	2D+time cube of mixed layer depths calculated by the density criteria where is the first density > 0.125 units from the surface (as caused by T and S criteria)
	# 	keywords: MLD mld mixed layer depth cmip5 mixed layer height
	##############################
	depth_cube = S[0].copy()
	depths = depth_cube.coord('depth').points
	#for memorys sake, do one year at a time..
	depth_cube.data = np.ma.masked_array(np.swapaxes(np.tile(depths,[360,180,1]),0,2))
	thetao = iris.load_cube(directory+model+'*_thetao_past1000_*Omon*.nc') 
	hm = S.extract(iris.Constraint(depth = np.min(depth_cube.data)))
	latitude_deg = depth_cube.copy()
	latitude_deg_data = hm.coord('latitude').points
	latitude_deg.data = np.swapaxes(np.tile(latitude_deg_data,[np.shape(S)[1],360,1]),1,2)
	hm.data = mld(S,thetao,depth_cube,latitude_deg)
	return hm


