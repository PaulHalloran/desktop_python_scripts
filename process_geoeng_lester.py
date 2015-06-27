import iris
import carbchem
import iris.quickplot as qplt
import matplotlib.pyplot as plt

directory = '/home/ph290/data0/lester_geoeng/'

files = ['anfea.pp','anfeb.pp','anfec.pp','anfed.pp']

for i,file in enumerate(files):
	cube = iris.load(directory+files[i])

	aragonite_sat_state = cube[2].copy()
	aragonite_sat_state.data = carbchem.carbchem(9,cube[0].data.fill_value,cube[2].data,cube[3].data*1000+35.0,cube[0].data/(1026.*1000.),cube[1].data/(1026.*1000.))

	iris.fileformats.netcdf.save(aragonite_sat_state, directory+files[i][0:5]+'_aragonite_sat_state.nc', netcdf_format='NETCDF3_CLASSIC')
	iris.fileformats.netcdf.save(cube[2], directory+files[i][0:5]+'_sst.nc', netcdf_format='NETCDF3_CLASSIC')

