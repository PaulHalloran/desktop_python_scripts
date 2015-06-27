#!/usr/bin/python2.7
'''
Module containing routines to work with ocean data.

AUTHOR:
    Chris Roberts (hadrr)

LAST MODIFIED:
    2013-07-15 - created (hadrr)

'''
import numpy as np

def eos_insitu(t,s,z):
    '''
    NAME:
        eos_insitu

    DESCRIPTION:
        Python version of in situ density calculation done by NEMO
        routine eos_insitu.f90. Computes the density referenced to
        a specified depth from potential temperature and salinity
        using the Jackett and McDougall (1994) equation of state.
        
    USAGE: 
        density = eos_insitu(T,S,p)

    INPUTS:
        T - potential temperature (celsius) 
        S - salinity              (psu)
        p - pressure              (dbar)
        
    OUTPUTS
        density - in situ density (kg/m3) - 1000.

    NOTES:
        Original routine returned (rho(t,s,p) - rho0)/rho0. 
        This version returns rho(t,s,p). Header for eos_insitu.f90
        included below for reference.

        ***  ROUTINE eos_insitu  ***
        
        ** Purpose :   Compute the in situ density from 
        potential temperature and salinity using an equation of state
        defined through the namelist parameter nn_eos. nn_eos = 0 
        the in situ density is computed directly as a function of
        potential temperature relative to the surface (the opa t
        variable), salt and pressure (assuming no pressure variation
        along geopotential surfaces, i.e. the pressure p in decibars
        is approximated by the depth in meters.
        
        ** Method  :  
        nn_eos = 0 : Jackett and McDougall (1994) equation of state.
        the in situ density is computed directly as a function of
        potential temperature relative to the surface (the opa t
        variable), salt and pressure (assuming no pressure variation
        along geopotential surfaces, i.e. the pressure p in decibars
        is approximated by the depth in meters.
        rho = eos_insitu(t,s,p)
        with pressure                 p        decibars
        potential temperature         t        deg celsius
        salinity                      s        psu
        reference volumic mass        rau0     kg/m**3
        in situ volumic mass          rho      kg/m**3
        
        Check value: rho = 1060.93298 kg/m**3 for p=10000 dbar,
        t = 40 deg celcius, s=40 psu
        
        References :   Jackett and McDougall, J. Atmos. Ocean. Tech., 1994
        
    AUTHOR:
        Chris Roberts (hadrr)

    LAST MODIFIED: 
        2013-08-15 - created (hadrr)
    '''
    # Convert to double precision
    ptem   = np.double(t)    # potential temperature (celcius)
    psal   = np.double(s)    # salintiy (psu)
    depth  = np.double(z)    # depth (m)
    rau0   = np.double(1035) # volumic mass of reference (kg/m3)
    # Read into eos_insitu.f90 varnames  
    zrau0r = 1.e0 / rau0
    zt     = ptem
    zs     = psal
    zh     = depth            
    zsr    = np.sqrt(np.abs(psal))   # square root salinity
    # compute volumic mass pure water at atm pressure
    zr1 = ( ( ( ( 6.536332e-9*zt-1.120083e-6 )*zt+1.001685e-4)*zt-9.095290e-3 )*zt+6.793952e-2 )*zt+999.842594
    # seawater volumic mass atm pressure
    zr2    = ( ( ( 5.3875e-9*zt-8.2467e-7 ) *zt+7.6438e-5 ) *zt-4.0899e-3 ) *zt+0.824493
    zr3    = ( -1.6546e-6*zt+1.0227e-4 ) *zt-5.72466e-3
    zr4    = 4.8314e-4
    #  potential volumic mass (reference to the surface)
    zrhop  = ( zr4*zs + zr3*zsr + zr2 ) *zs + zr1
    # add the compression terms
    ze     = ( -3.508914e-8*zt-1.248266e-8 ) *zt-2.595994e-6
    zbw    = (  1.296821e-6*zt-5.782165e-9 ) *zt+1.045941e-4
    zb     = zbw + ze * zs
    zd     = -2.042967e-2
    zc     =   (-7.267926e-5*zt+2.598241e-3 ) *zt+0.1571896
    zaw    = ( ( 5.939910e-6*zt+2.512549e-3 ) *zt-0.1028859 ) *zt - 4.721788
    za     = ( zd*zsr + zc ) *zs + zaw
    zb1    =   (-0.1909078*zt+7.390729 ) *zt-55.87545
    za1    = ( ( 2.326469e-3*zt+1.553190)*zt-65.00517 ) *zt+1044.077
    zkw    = ( ( (-1.361629e-4*zt-1.852732e-2 ) *zt-30.41638 ) *zt + 2098.925 ) *zt+190925.6
    zk0    = ( zb1*zsr + za1 )*zs + zkw
    # Caculate density
    prd    = (  zrhop / (  1.0 - zh / ( zk0 - zh * ( za - zh * zb ) )  ) - rau0  ) * zrau0r
    rho    = (prd*rau0) + rau0
    return rho - 1000.


def calc_dens(t,s,unesco=False):
    '''
    NAME:
       calc_dens
    
    DESCRIPTION:
        Python version of TIDL calc_dens.pro  routine for calculating
        sea water potential density:

        /usr/local/tidl8/cr/lib/calc_dens.pro

    USAGE: 
        density = calc_dens(T,S,[unesco=False])

    INPUTS:
        T - potential temperature (celsius) 
        S - salinity              (psu)

    KEYWORDS:
        unesco - If specified, approximate UNESCO eqn of
                 state used. Default is Knudsen.
                
    OUTPUTS
        density - sea water potential density (kg/m3)
            
    AUTHOR:
        Chris Roberts (hadrr)

    LAST MODIFIED: 
        2013-08-15 - created (hadrr)
    '''
    # Define cooefficients
    if unesco:
        TO = 13.4993292
        SO = -0.0022500
        SIGO = 24.573975
        C = [-0.2016022E-03,0.7730564E+00,-0.4919295E-05,-0.2022079E-02,
              0.3168986E+00,0.3610338E-07, 0.3777372E-02, 0.3603786E-04,
              0.1609520E+01]
    else:
        # Knudsen values
        TO = 13.4992332
        SO = -0.0022500
        SIGO = 24.573651
        C = [-.2017283E-03,0.7710054E+00,-.4918879E-05,-.2008622E-02,
               0.4495550E+00,0.3656148E-07,0.4729278E-02,0.3770145E-04,
               0.6552727E+01]
    # Calc potential density 
    TQ = t -TO
    SQ = (s-35.)/1000. - SO
    dens = (C[0]+(C[3]+C[6]*SQ)*SQ+(C[2]+C[7]*SQ+C[5]*TQ)*TQ)*TQ+(C[1]+(C[4]+C[8]*SQ)*SQ)*SQ
    return dens*1000.+ SIGO
    
