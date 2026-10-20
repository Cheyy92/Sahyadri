import sys
sys.path.append("/mnt/home/project/csatsangi.chetan/projects/modified_poisson")
sys.path.append("/mnt/home/project/csatsangi.chetan/projects/spectrophotogroup/Simulator")
sys.path.append("/mnt/home/project/csatsangi.chetan/projects/sahyadri-sandbox/scripts/post-process")
# import simulator
import readers
from readers import SnapshotReader
from readers import HaloReader
import numpy as np
np.int = int # Add this line to bypass the error
import scipy.special as sysp 
import scipy.spatial as syspat
import pandas as pd
import gc
import simulator
from astropy.table import Table


#Fitting functions for HOD Parameters:
def _M_function(Mr, A, B, C, D):
    x = Mr + 20
    return (A + 12) + B*x + C*x**2 + D*x**3          # returns log10(M)

def _M0_function(Mr, A, B):
    x = Mr + 20
    logM0 = (A + 11) + B*x
    return np.maximum(logM0, 1.0)                            # scalar-safe floor

def _sigma_function(Mr, A, B, C, D):
    x = Mr + 20
    return A + (B - A) / (1. + np.exp(C*(x + D)))

def _alpha_function(Mr, A, B, C):
    x = Mr + 20
    return A + B**(-x + C)

def hod_params_func(Mr):
    """Smith, Grove, Cole et al. (2024) nested BGS HOD, AbacusSummit c000/ph000 best fit.
    Returns Mcut, M1, sigma, alpha, kappa (kappa*Mcut == M0) for simulator.py's HOD_model.
    """
    logMmin = _M_function(Mr, _A_min, _B_min, _C_min, _D_min)
    logM1   = _M_function(Mr, _A_1,   _B_1,   _C_1,   _D_1)
    logM0   = _M0_function(Mr, _A_0, _B_0)
    sigma   = _sigma_function(Mr, _A_sig, _B_sig, _C_sig, _D_sig)
    alpha   = _alpha_function(Mr, _A_al, _B_al, _C_al)
    
    Mcut = 10**logMmin
    M1   = 10**logM1
    M0   = 10**logM0
    kappa = M0 / Mcut

    return dict(Mcut=Mcut, M1=M1, sigma=sigma, alpha=alpha, kappa=kappa)

#///////////////////////////////////////////////////////////////////
# HOD Fits
#///////////////////////////////////////////////////////////////////
class HODFits_BGS(object):
    """ Fitting functions and derivatives specifying best fit 5-parameter HOD fits 
         from mith, Grove, Cole et al. (2024) Public Data Release.
    """
    def __init__(self):
        """ Best fit HOD parameters and derivatives as function of luminosity threshold Mr. """
    
        self._A_min, self._B_min, self._C_min, self._D_min = -0.1163295942035853792, -0.5759066407274642252, 0.1437085418067726994, -0.01908335449679073731
        self._A_sig, self._B_sig, self._C_sig, self._D_sig = 0.01832666435246161490, 0.7584865035087475782, 1.103103765754579246, 0.3277787214124110449
        self._A_0,   self._B_0                    = -0.5468895455660652827, -1.673562380257160864
        self._A_1,   self._B_1, self._C_1, self._D_1         = 1.121497360473139082, -0.4725625165007949491, 0.09200038944813740405, -0.01141762462160326314
        self._A_al,  self._B_al, self._C_al             = 1.111866664500087198, 4.473949866395862784, -4.353309870091147893


    #Fitting functions for HOD Parameters:
    def _M_function(Mr, A, B, C, D):
        x = Mr + 20
        return (A + 12) + B*x + C*x**2 + D*x**3          # returns log10(M)

    def _M0_function(Mr, A, B):
        x = Mr + 20
        logM0 = (A + 11) + B*x
        return np.maximum(logM0, 1.0)                            # scalar-safe floor

    def _sigma_function(Mr, A, B, C, D):
        x = Mr + 20
        return A + (B - A) / (1. + np.exp(C*(x + D)))

    def _alpha_function(Mr, A, B, C):
        x = Mr + 20
        return A + B**(-x + C)

    def hod_fit(self, param, Mr):
        """Smith, Grove, Cole et al. (2024) nested BGS HOD, AbacusSummit c000/ph000 best fit.
        Returns Mcut, M1, sigma, alpha, kappa (kappa*Mcut == M0) for simulator.py's HOD_model.
        """
        if param == 'kappa':
            lgMmin = _M_function(Mr, self._A_min, self._B_min, self._C_min, self._D_min)
            Mcut = 10**lgMmin
            logM0   = _M0_function(Mr, self._A_0, self._B_0)
            M0   = 10**logM0
            kappa = M0 / Mcut
            return kappa
        elif param == 'M1':
            logM1   = _M_function(Mr, self._A_1, self._B_1, self._C_1, self._D_1)
            M1   = 10**logM1
            return M1
        elif param == 'siglgM':
            sigma   = _sigma_function(Mr, self._A_sig, self._B_sig, self._C_sig, self._D_sig)
            return sigma
        elif param == 'alpha':
            alpha   = _alpha_function(Mr, self._A_al, self._B_al, self._C_al)
            return alpha
        elif param == 'Mcut':
            lgMmin = _M_function(Mr, self._A_min, self._B_min, self._C_min, self._D_min)
            Mcut = 10**lgMmin
            return Mcut
        elif param == 'M0':
            logM0   = _M0_function(Mr, self._A_0, self._B_0)
            M0   = 10**logM0
            return M0
        elif param == 'prs':
            # From the Paul et al, 2019 doi:10.1093/mnras/stz1764 # Assuming the SDSS colour assignment works for DESI BGS catalog too
            a0,a1,a2 = 0.773, -0.065, 0.008
            x = Mr + 20.5
            out = (a0 + a1*x + a2*x**2)
            return out

    

#///////////////////////////////////////////////////////////////////
# HOD Functions
#///////////////////////////////////////////////////////////////////
class HODFunctions(HODFits_BGS):
    """ Functions and derivatives specifying 5-parameter HODs 
         Uses fits from Smith et al, 2024.
    """
    def __init__(self):
        """ HOD functions and derivatives as function of luminosity threshold Mr. """

        HODFits_BGS.__init__(self)
        
    def F_spline(self,x):
        """
        The Cumulative integral function as defined in arXiv:2312.08792v2 
        The integral was analytically calculated using the spline function defined in the paper.
        """
        t = x / np.sqrt(6)
        at = np.abs(t)
        sign = np.sign(x)

        piece1 = (8/3)*t - (16/3)*t**3 + 4*t**4
        piece2 = np.sign(t) * (1 - (4/3)*(1 - at)**4)
        piece3 = np.sign(t) * 1.0

        result = np.where(
            at <= 0.5, piece1,
            np.where(at <= 1, piece2, piece3)
        )
        return result
    
        
    def F_arg(self,Mr,lgm):
        """ Convenience function.
             If both arguments are arrays, output has shape (lgm.size,Mr.size).
        """
        Mr_sc = np.isscalar(Mr)
        lgm_sc = np.isscalar(lgm)

        lgm_arr = np.outer(lgm,np.ones(Mr.size)) if ((not Mr_sc) & (not lgm_sc)) else 1.0*lgm
        arg = (lgm_arr - np.log10(self.hod_fit('Mcut',Mr)))/self.hod_fit('siglgM',Mr)
        return arg

    def erf_arg(self,Mr,lgm):
        """ Convenience function.
             If both arguments are arrays, output has shape (lgm.size,Mr.size).
        """
        Mr_sc = np.isscalar(Mr)
        lgm_sc = np.isscalar(lgm)

        lgm_arr = np.outer(lgm,np.ones(Mr.size)) if ((not Mr_sc) & (not lgm_sc)) else 1.0*lgm
        erfarg = (lgm_arr - np.log10(self.hod_fit('Mcut',Mr)))/self.hod_fit('siglgM',Mr)
        return erfarg

    def fcen_thresh(self,Mr,lgm,hod):
        """ fcen(>L|m) = (1/2)[1 + erf(log10(m/Mmin)/siglgM)]. 
             If both arguments are arrays, output has shape (lgm.size,Mr.size).
        """
        if(hod=='erf'):
            erfarg = self.erf_arg(Mr,lgm)
            out = 0.5*(1+sysp.erf(erfarg))
            return out
        elif(hod=='BGS'):
            Farg = self.F_arg(Mr, lgm)
            out = 0.5*(1+self.F_spline(Farg))
            return out

    def Nsat_thresh(self,Mr,lgm):
        """ Nsat(>L|m) = ((m - M0)/M1)^alpha.
             If both arguments are arrays, output has shape (lgm.size,Mr.size).
        """
        M0 = self.hod_fit('M0',Mr)
        M1 = self.hod_fit('M1',Mr)
        alpha = self.hod_fit('alpha',Mr)
        mhalo = 10**lgm

        Mr_sc = np.isscalar(Mr)
        lgm_sc = np.isscalar(lgm)

        if Mr_sc & lgm_sc:
            out = ((mhalo - M0)/M1)**alpha if mhalo > M0 else 0.0
        else:
            mhalo_arr = np.outer(mhalo,np.ones(Mr.size)) if ((not Mr_sc) & (not lgm_sc)) else 1.0*mhalo
            Dm = (mhalo_arr - M0)/M1
            Dm[Dm < 0.0] = 0.0
            out = Dm**alpha

        return out
    
    def mean_color_sat(self, Mr):
        return 0.83 - 0.08*(Mr + 20)
    
    def mean_color_red(self, Mr):
        return 0.932 - 0.032*(Mr + 20)
    
    def std_color_red(self, Mr):
        return 0.07 + 0.01*(Mr+20)
    
    def mean_color_blue(self, Mr):
        return 0.62 - 0.11*(Mr+20)
    
    def std_color_blue(self, Mr):
        return 0.12 + 0.02*(Mr+20)
    
    def blue_frac(self, Mr):
        return 0.46 + 0.07*(Mr+20)
    
    def p_red_sat(self, Mr):
        num = self.mean_color_sat(Mr) - self.mean_color_blue(Mr)
        den = self.mean_color_red(Mr) - self.mean_color_blue(Mr)
        return num/den


class Mocker(HODFunctions):
    def __init__(self, hod):
#         self.real = real
#         self.snap = snap
#         self.sim_stem = sim_stem
        
#         print("Initializing SnapshotReader...")
#         sr = SnapshotReader(sim_stem=self.sim_stem, real=self.real, snap=self.snap, use_compressed=True)
        
        HODFunctions.__init__(self)
        # self.redshift = sr.redshift
        # self.Om = sr.Om
        # self.Lbox = sr.Lbox # Mpc/h
        # self.massdef = 'm200b'
        # self.mmin = 2e11 # Msun/h
        self.Mrmax = -18.0 #Maximum(faintest) r-band absolute magnitude for galaxies
        self.hod = hod
#         self.Dvir = 200*self.Om #Virial density threshold
        
#         self.rhoc = 2.7754e11 # (Msun/h) / (Mpc/h)^3 - Critical Density pf the universe at z=0
#         self.rhoc_z = self.rhoc*self.EHub(self.redshift)**2 #Critical density at a particular redshift
        
        self.rng = np.random.RandomState(seed=42) #Random number generator

        dMr = 0.001 #Bin spacing for Mrvals
        nMr = int((23.5+self.Mrmax)/dMr) #number of Mrvals
        self.Mrvals = np.linspace(-23.5,self.Mrmax,nMr) # useful for interpolation later
        self.dMr = self.Mrvals[1]-self.Mrvals[0]
        
    def assign_central_luminosity(self,lgm_halos):
        """ Assign central luminosities. Can be generalised to depend on more than halo mass. """
        
        print("... luminosities")
        Mr_cen = np.zeros(lgm_halos.size,dtype=float)

        ############################################
        # Generate uniform deviates
        u = self.rng.rand(lgm_halos.size)

        ################################################
        for h in range(lgm_halos.size):
            # Luminosity function is Pcen(>L|m) = fcen(>L|m)/fcen(>Lmin|m)
            PcenL = self.fcen_thresh(self.Mrvals,lgm_halos[h], self.hod)/self.fcen_thresh(self.Mrmax,lgm_halos[h], self.hod)
            # keeping this in loop is not much slower than vectorised calc, but more memory-efficient

            #finding the bound of u :
            # Find largest L (smallest Mr) where Pcen(>L|m) > u
            Mr_upp = self.Mrvals[PcenL > u[h]][0]

            # Find smallest L (largest Mr) where Pcen(>L|m) <= u
            Mr_low = self.Mrvals[PcenL <= u[h]][-1]
            # choose fainter of the two (enforce nesting of HOD) : A halo capable of holding a bright galaxy must dynamically be capable of holding a fainter one
            Mr_cen[h] = np.max([Mr_upp,Mr_low])
        ################################################
        
        del u
        gc.collect()
        
        print('done!')
        return Mr_cen
    ########################################################
    
    ########################################################
    def assign_satellite_luminosities(self,centrals,h):
        """ Assign satellite luminosities. """

        Nsat_h = centrals['Nsat'][h]
        Mr_cen = centrals['Mr'][h]
        lgmhalo = centrals['lgm'][h]
        PsatL = self.Nsat_thresh(self.Mrvals,lgmhalo)/self.Nsat_thresh(self.Mrmax,lgmhalo)

        ####################################################
        # Generate luminosities, without worrying about Lcen.
        # ... generate Nsat_h uniform deviates
        u = self.rng.rand(Nsat_h)
        # ... find largest L (smallest Mr) where Psat(>L|m) > u
        Mr_sat = np.array([self.Mrvals[(PsatL >= u[s])][0] for s in range(Nsat_h)])
        return Mr_sat
    ########################################################
    
    def assign_centrals_colors(self, centrals, satellites, Mr_bin_edges = [-23, -22, -21, -20, -19, -18]):
        color = np.empty(centrals.size, dtype ='U4')
        u = self.rng.rand(centrals.size)
        
        #From Zehavi et al(SDSS), 2011
        p_red_bin = {
            (-23, -22): 0.82,
            (-22, -21): 0.63,
            (-21, -20): 0.53,
            (-20, -19): 0.43,
            (-19, -18): 0.28
        }
        
        for i in range(len(Mr_bin_edges)-1):
            Mr_hi, Mr_lo = Mr_bin_edges[i], Mr_bin_edges[i+1]
            bin_key = (Mr_hi, Mr_lo)
            p_red = p_red_bin.get(bin_key)
            
            #Fractions:
            in_bin_cen = (centrals['Mr'] >= Mr_hi) & (centrals['Mr'] < Mr_lo)
            in_bin_sat = (satellites['Mr'] >= Mr_hi) & (satellites['Mr'] < Mr_lo)
            n_cen, n_sat = in_bin_cen.sum(), in_bin_sat.sum()
            p_cen_bar = n_cen/(n_cen+n_sat)
            p_sat_bar = n_sat/(n_cen+n_sat)
            
            #prs:
            prs_per_gal = np.clip(self.hod_fit('prs', centrals['Mr'][in_bin_cen]), 0.01, 0.99)
            
            #Eq A.10 from paul,pahwa,paranjape(2019):
            p_red_cen  =(p_red - prs_per_gal*p_sat_bar)/p_cen_bar
            is_red = u[in_bin_cen] < p_red_cen
            color[in_bin_cen] = np.where(is_red, 'red', 'blue')
            
        return color
    
    ########################################################
    def assign_satellite_colors(self, satellites):
        """ Red/blue assignment for satellites using prs(Mr), indipendent of halo mass"""
        
        prs = self.hod_fit('prs', satellites['Mr'])
        prs = np.clip(prs, 0.01, 0.99)
        u = self.rng.rand(satellites.size)
        is_red = u < prs
        color = np.where(is_red, 'red', 'blue')

        return color
    ########################################################
    
    def assign_satellite_gr(self, satellites, add_scatter=False):
        u = self.rng.uniform(0,1, satellites.shape[0])
        is_red = u < self.p_red_sat(satellites['Mr'])
        gr_color = np.zeros(satellites['Mr'].shape[0])
        if np.any(is_red):
            mr_red = satellites['Mr'][is_red]
            mean_red = self.mean_color_red(satellites['Mr'][is_red])
            if add_scatter:
                sigma_red = self.std_color_red(mr_red) # Assuming a scatter function exists
                mean_red += self.rng.normal(0, sigma_red)    
            gr_color[is_red] = mean_red
        
        if np.any(~is_red):
            mr_blue = satellites['Mr'][~is_red]
            mean_blue = self.mean_color_red(satellites['Mr'][~is_red])
            if add_scatter:
                sigma_blue = self.std_color_blue(satellites['Mr'][~is_red])
                mean_blue += self.rng.normal(0, sigma_blue)
            gr_color[~is_red] = mean_blue
        
        return gr_color
        
    
    ########################################################
    
    def assign_central_gr(self, centrals, nsat, add_scatter=False):
        
        u = self.rng.uniform(0,1, centrals.shape[0])
        ncen = centrals['Mr'].shape[0]
        cen_frac = 1/(1+nsat/ncen)
        is_red = u > self.blue_frac(centrals['Mr'])/cen_frac
        gr_color = np.zeros(ncen)
        
        if np.any(is_red):
            mr_red = centrals['Mr'][is_red]
            mean_red = self.mean_color_red(centrals['Mr'][is_red])
            if add_scatter:
                sigma_red = self.std_color_red(mr_red) # Assuming a scatter function exists
                mean_red += self.rng.normal(0, sigma_red)    
            gr_color[is_red] = mean_red
          
        if np.any(~is_red):
            mr_blue = centrals['Mr'][~is_red]
            mean_blue = self.mean_color_red(centrals['Mr'][~is_red])
            if add_scatter:
                sigma_blue = self.std_color_blue(centrals['Mr'][~is_red])
                mean_blue += self.rng.normal(0, sigma_blue)
            gr_color[~is_red] = mean_blue
        
        return gr_color
        



#Cosmology:
redshift= 0.2
scale = 1/(1+redshift) #look up scales.txt in /data/project/hpc2502016/data/halos/sahyadri/default2048 for snapshot

#Sahyadri Params:
mpart_sahyadri = 81084619.46190532
mmin = 40*mpart_sahyadri

#Halo Cat if using simulator:
var = sys.argv[1]
halo_cat = f'/data/project/hpc2502016/data/halos/sahyadri/{var}/r1/compressed/out_82_basic.fits.gz'

#Using HaloReader
real = 1
snap = 82
sim_stem = "sahyadri/default2048"

# best-fit meta-parameters. Refer to arXiv:2312.08792v2 (DESI BGS)

_A_min, _B_min, _C_min, _D_min = -0.1163295942035853792, -0.5759066407274642252, 0.1437085418067726994, -0.01908335449679073731
_A_sig, _B_sig, _C_sig, _D_sig = 0.01832666435246161490, 0.7584865035087475782, 1.103103765754579246, 0.3277787214124110449
_A_0,   _B_0                    = -0.5468895455660652827, -1.673562380257160864
_A_1,   _B_1, _C_1, _D_1         = 1.121497360473139082, -0.4725625165007949491, 0.09200038944813740405, -0.01141762462160326314
_A_al,  _B_al, _C_al             = 1.111866664500087198, 4.473949866395862784, -4.353309870091147893

#Hod Params
hod_params = hod_params_func(-18.0) #since the BGS paper works with this number

random_seed=10

halo_ftype='fits_Bolshoi'
central_model='BGS'
satellite_model='power_law'
angular_distribution='spherical'
radial_profile='NFW'
los='z'
geometry='cubic'
z_cos = 0.2
Mcut=hod_params['Mcut']
M1=hod_params['M1']
sigma=hod_params['sigma']
pmax =1.0
alpha=hod_params['alpha']
kappa=hod_params['kappa']
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

#Simulator
#note: set rsd to false if needed
OneH=simulator.default_mock(halo_file=halo_cat,halo_ftype=halo_ftype,
        constant_nsat=constant_nsat,central_model=central_model,satellite_model=satellite_model,
        Mcut=Mcut,M1=M1,sigma = sigma,pmax =pmax,alpha=alpha, kappa=kappa,
        alpha_einasto= alpha_einasto, Ap = Ap,
        fradsat=fradsat,fconc=fconc,gNFW=gNFW,Q=Q,gamma=gamma,                 
        angular_distribution=angular_distribution,radial_profile=radial_profile,
        los=los,geometry=geometry, z_cos = z_cos,verbose=verbose,
        return_full_obj=return_full_obj,random_seed=random_seed,
        rsd=True,gravz=False)

#initialize the mocker
mm = Mocker(hod='BGS')
halos = {key: val[OneH.cen_halo_index] for key, val in OneH.halos.halos.items()} #Has the halo data from simulator

#We need combined data of positions + luminosities for centrals:
centrals = np.zeros(halos['mass'].size,dtype=[('haloid','int'),
                                      ('lgm','double'),('rvir','double'),('con','double'),
                                      ('x','double'),('y','double'),('z','double'),
                                      ('Mr','double'),('Nsat','int'),('color', 'U4'), ('g-r', 'double')]) #EXTRA COLUMNS FOR FUTURE USE
centrals['x'], centrals['y'], centrals['z'] = OneH.cen_xyz.T
centrals['lgm'] = np.log10(halos['mass'])
centrals['haloid'] = OneH.cen_halo_index
centrals['Nsat'] = OneH.hod.nsat[OneH.cen_halo_index]

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

#Assign Luminosities for centrals
centrals['Mr'] = mm.assign_central_luminosity(centrals['lgm'])

#Assign Luminosity for satellites
s_lo = 0
print('...satellite luminosities')
for h in range(centrals.size):
    s_hi = s_lo + centrals['Nsat'][h]
    ################################################
    if centrals['Nsat'][h]:
        ############################################
        sMr = mm.assign_satellite_luminosities(centrals,h)
        satellites['Mr'][s_lo:s_hi] = sMr
    ################################################
    # Update
    s_lo = s_hi
print('done!')

Table(centrals).write('centrals_rsd.fits', overwrite=True)
Table(satellites).write('satellites_rsd.fits', overwrite=True)

