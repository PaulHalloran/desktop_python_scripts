'''
Run model by typing: 'python2.7 bolton_stoll.py' into the terminal in an Apple Mac or a LINUX machine.

'''

import numpy as np

'''

*********************************************************************
*																	*
*		MODIFY THE VALUES IN THIS SECTION, SAVE AND RUN MODEL		*
*																	*
*********************************************************************

All fluxes are in units of 10-17 mol s-1

Values to play with:
'''

# FCiu =	1.0 # CO2 flux from extracellular media to cell cytosol
# FBiu =	1.0 # HCO3- flux from extracellular media into cell cytosol
# FCxu =	1.0 # CO2 flux from cytosol to chloroplast
# FBxu =	1.0 # HCO3- flux from cytosol to chloroplast
# FCio =	1.0 # CO2 flux from cell cytosol to extracellular media
# Ffix =	1.0 # Photosynthetic carbon fixation flux
# FHydi =	1.0 # dehydration of HCO3- to CO2 in cell cytosol
# FDehi =	1.0 # hydration of CO2 to HCO3- in cell cytosol
# FHydx =	1.0 # dehydration of HCO3- to CO2 in chloroplast
# FDehx =	1.0 # dehydration of HCO3- to CO2 in chloroplast
# FCvu =	1.0 # CO2 flux from cytosol to coccolith vesicle
# FBvu =	1.0 # HCO3- flux from cytosol to coccolith vesicle
# FHydv =	1.0 # hydration of CO2 to HCO3- in coccolith vesicle
# FDehv =	1.0 # dehydration of HCO3- to CO2 in coccolith vesicle
# Fcal = 1.0 # Calcification flux

f = open('./bolton_stoll_input_data.txt')
lines = f.readlines()
f.close()

tmp = np.empty(15)
for i,line in enumerate(lines):
	 tmp[i] = lines[0].split('=')[1].split(' ')[0]
	 
FCiu =	tmp[0]
FBiu =	tmp[1]
FCxu =	tmp[2]
FBxu =	tmp[3]
FCio =	tmp[4]
Ffix =	tmp[5]
FHydi =	tmp[6]
FDehi =	tmp[7]
FHydx =	tmp[8]
FDehx =	tmp[9]
FCvu =	tmp[10]
FBvu =	tmp[11]
FHydv =	tmp[12]
FDehv =	tmp[13]
Fcal = tmp[14]

'''
Table S1: Notation used in numerical model 
All fluxes are given in units of 10-17 mol s-1
'''

FCxo =	0.01 # DEFAULT BVALUE SPECIFIED IN TABLE S1 CO2 flux from chloroplast to cytosol
FBxo =	0.0 # DEFAULT BVALUE SPECIFIED IN TABLE S1 HCO3- flux from chloroplast to cytosol
FBio =	0.0 # DEFAULT BVALUE SPECIFIED IN TABLE S1 HCO3- flux from cell cytosol to extracellular media
FCvo =	0.0 # DEFAULT BVALUE SPECIFIED IN TABLE S1 CO2 flux from coccolith vesicle to cytosol
FBvo =	0.0 # DEFAULT BVALUE SPECIFIED IN TABLE S1 HCO3- flux from coccolith vesicle to cytosol

'''
The following are not used, but inform your parameter choice
'''

fcv = 0.4 # Ratio of effective permeability of coccolith vesicle to that of plasma membrane
	# range: 0.1 -0.4
fchl = 0.02 # Ratio of effective permeability of chloroplast to that of plasma membrane
	# range: 0.02-0.10


'''
Table S2: Fractionation factors applied in numerical model
values are o/oo
'''


Et = -10.0 # Thermodynamic fractionation of CO2 relative to HCO3
Ebc = -10.0 # CA-catalysed dehydration of HCO3
Ecb = -1.0 # CA-catalysed hydration of CO2
Ef = -27.0 # fractionation during CO2 fixation by Rubisco
Ecal = 1.0 # fractionation during CaCO3 precipitation from HCO3

'''
Carbon isotope values
'''

dCe = 1.0 #  d13C of CO2 in external media
dBe = 1.0 # d13C HCO3- in external media

'''
Solve to find:
'''

# dCi #  d13C of CO2 in cell cytosol
# dCx #  d13C of CO2 in chloroplast
# dBi #  d13C HCO3- in cell cytosol
# dBx #  d13C HCO3- in chloroplast
# dCv #  d13C of CO2 in coccolith vesicle
# dBv #  d13C of HCO3- in coccolith vesicle


'''
Equations:
(1) Chloroplast CO2
(2) Chloroplast HCO3-
(3) Coccolith vesicle CO2
(4) Coccolith vesicle HCO
(5) Cytosol CO2
(6) Cytosol HCO3
'''

#the following are formulated from the equations in the Bolton and Stoll supp. Mat.

#equations:1				 	2					3					 4						 5						 		6	
dCi = [FCxu						,0					,FCvu				,0						,(-1.0)*FCio-FCxu-FCvu-FHydi	,FHydi]
dCx = [(-1.0)*FCxo-FHydx-Ffix	,FHydx				,0					,0						,FCxo							,0]
dBi = [0						,FBxu				,0					,FBvu					,FDehi-FCio-FCxu-FCvu			,(-1.0)*FBio-FBxu-FBvu-FDehi]
dBx = [FDehx					,(-1.0)*FBxo-FDehx	,0					,0						,FBxo							,FBxo]
dCv = [0						,0					,(-1.0)*FCvo-FHydv	,FHydv					,FCvo							,0]
dBv = [0						,0					,FDehv				,(-1.0)*FBvo-FDehv-Fcal	,FBvo							,FBvo]

b = np.array([(-1.0)*(FDehx*Ebc)+FHydx*Ecb+Ffix*Ef,
			FHydx*Ecb+FDehx*Ebc,
			FDehv*Ebc+FHydv*Ecb,
			FHydv*Ecb+FDehv*Ebc+Fcal*Ecal,
			(-1.0)*dCe*FCiu-FDehi*Ebc+FHydi*Ecb,
			dBe*FBiu-FHydi*Ecb+FDehi*Ebc])


A = np.array([[dCi[0],dCx[0],dBi[0],dBx[0],dCv[0],dBv[0]],
	[dCi[1],dCx[1],dBi[1],dBx[1],dCv[1],dBv[1]],
	[dCi[2],dCx[2],dBi[2],dBx[2],dCv[2],dBv[2]],
	[dCi[3],dCx[3],dBi[3],dBx[3],dCv[3],dBv[3]],
	[dCi[4],dCx[4],dBi[4],dBx[4],dCv[4],dBv[4]],
	[dCi[5],dCx[5],dBi[5],dBx[5],dCv[5],dBv[5]]])


x = np.linalg.solve(A, b)
#Solving above equation matrices using linear algebra

print 'd13C of CO2 in cell cytosol (dCi) = '+np.str(x[0]) 
print 'd13C of CO2 in chloroplast (dCx) = '+np.str(x[1])
print 'd13C HCO3- in cell cytosol (dBi) = '+np.str(x[2])
print 'd13C HCO3- in chloroplast (dbx) = '+np.str(x[3])
print 'd13C of CO2 in coccolith vesicle (dCv)'+np.str(x[4])
print 'd13C of HCO3- in coccolith vesicle (dBv)'+np.str(x[5])
print 'units = per mil'


# print x
# print np.dot(A,x)

