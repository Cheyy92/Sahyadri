#Code to generate multiple catalog with varied parameters

import os
# Use the path where readers.py is supposed to be
root = "/mnt/home/project/csatsangi.chetan/projects"
import sys
sys.path.insert(0, os.path.join(root, "sahyadri-sandbox", "scripts", "post-process"))
sys.path.insert(0, "/mnt/home/project/csatsangi.chetan/projects/modified_poisson")
sys.path.insert(0, "/mnt/home/project/csatsangi.chetan/projects/spectrophotogroup/Simulator")
sys.path.insert(0, "/mnt/home/project/csatsangi.chetan/projects/sahyadri-sandbox/scripts/post-process")
sys.path.insert(0, "/mnt/home/project/csatsangi.chetan/projects/Derivatives/hod_utils")
import simulator
import numpy as np
from hod_utils import HODFits_BGS, Mocker
from astropy.table import Table
np.int=int


hod_fits = HODFits_BGS()
params = hod_fits.get_hod_params_dict(-18.0)
variation = sys.argv[1]
head = variation

#simulator parameters:
halo_cat = f'/data/project/hpc2502016/data/halos/sahyadri/{variation}/r1/compressed/out_82_basic.fits.gz'
random_seed=10
halo_ftype='fits_Bolshoi'
central_model='BGS'
satellite_model='power_law'
angular_distribution='spherical'
radial_profile='NFW'
los='z'
geometry='cubic'
z_cos = 0.2
pmax =1.0
verbose=True
return_full_obj=True
#parameter defined but not used with this setting
Ap = 1.0
fradsat=1.0
fconc=1.0
gNFW=1.0
Q=None
gamma=4.12 
alpha_einasto=0.16
constant_nsat=2

# PLUS VARIATION:
#HOD parameters:
print(f"Setting HOD parameters for {variation}...")
Mcut  = params['Mcut']
M1    = params['M1']
sigma = params['sigma']
alpha = params['alpha']
kappa = params['kappa']

#Occupy galaxies:
OneH=simulator.default_mock(halo_file=halo_cat,halo_ftype=halo_ftype,
        constant_nsat=constant_nsat,central_model=central_model,satellite_model=satellite_model,
        Mcut=Mcut,M1=M1,sigma = sigma,pmax =pmax,alpha=alpha, kappa=kappa,
        alpha_einasto= alpha_einasto, Ap = Ap,
        fradsat=fradsat,fconc=fconc,gNFW=gNFW,Q=Q,gamma=gamma,                 
        angular_distribution=angular_distribution,radial_profile=radial_profile,
        los=los,geometry=geometry, z_cos = z_cos,verbose=verbose,
        return_full_obj=return_full_obj,random_seed=random_seed,
        rsd=True,gravz=False)

halos = {key: val[OneH.cen_halo_index] for key, val in OneH.halos.halos.items()}

#We need combined data of positions + luminosities for centrals:
centrals = np.zeros(halos['mass'].size,dtype=[('haloid','int'),
                                      ('lgm','double'),('rvir','double'),('con','double'),
                                      ('x','double'),('y','double'),('z','double'),
                                      ('Mr','double'),('Nsat','int'),('color', 'U4'), ('g-r', 'double')]) #EXTRA COLUMNS FOR FUTURE USE
centrals['x'], centrals['y'], centrals['z'] = OneH.cen_xyz.T
centrals['lgm'] = np.log10(halos['mass'])
centrals['haloid'] = OneH.cen_halo_index
centrals['Nsat'] = OneH.hod.nsat[OneH.cen_halo_index]
Table(centrals).write(f'{head}/centrals_rsd.fits', overwrite=True)
print("Centrals Ready!")

# Same thing for satellites
Nsat_tot = np.sum(centrals['Nsat']) #Total Number of satellites to be assigned
# Allocate memory for satellites
satellites = np.zeros(Nsat_tot,dtype=[('haloid','int'),
                                      ('lgm','double'),('con','double'),
                                      ('x','double'),('y','double'),('z','double'),
                                      ('Mr','double'), ('color', 'U4'), ('g-r', 'double')])
satellites['haloid'] = OneH.sat_halo_index
satellites['x'], satellites['y'], satellites['z'] = OneH.sat_xyz.T
satellites['lgm'] = np.log10(OneH.halos.halos['mass'][OneH.sat_halo_index.astype(int)])
Table(satellites).write(f'{head}/satellites_rsd.fits', overwrite=True)
print("Satellites Ready!")