import sys
sys.path.append("/mnt/home/project/csatsangi.chetan/projects/modified_poisson")
sys.path.append("spectrophotogroup/Simulator")
sys.path.append("sahyadri-sandbox/scripts/post-process")
import numpy as np
import gc
import matplotlib
matplotlib.use('Agg')  # no display on compute node
import matplotlib.pyplot as plt
from astropy.table import Table
from numpy.lib import recfunctions as rfn
from voronoi_web import Voronoi
from scipy.spatial import cKDTree

np.int=int

#Load data:
centrals = Table.read('centrals.fits').as_array()
satellites = Table.read('satellites.fits').as_array()
centrals_trimmed = rfn.drop_fields(centrals, ('Nsat','rvir'), usemask=False)
all_galaxies = np.concatenate([centrals_trimmed, satellites])

gpos = np.zeros((all_galaxies.shape[0], 3), dtype='float')
gpos[:,0] = all_galaxies['x']
gpos[:,1] = all_galaxies['y']
gpos[:,2] = all_galaxies['z']

Ran_Fac_test = 1000
seed = 10

#voronoi_web
real = 1
snap = 82
sim_stem = "sahyadri/default2048"

v = Voronoi(sim_stem=sim_stem, real=real, snap=snap, seed=seed, verbose=True, N_Proc=10)

gpos = gpos%v.Lbox
pos = gpos
Lbox = v.Lbox
Ntrc = pos.shape[0]

Ran_Fac = Ran_Fac_test
v.Ran_Fac = Ran_Fac

delta_class = v.voronoi_periodic_box(pos, ret_ran=False)
y_class = 1.0/(1.0+delta_class)

#Using cKDTree:
rng = np.random.RandomState(seed)
Nran = Ran_Fac * Ntrc
ran_pos = Lbox * rng.rand(Nran, 3)

tree = cKDTree(gpos, boxsize=Lbox)
_, idx = tree.query(ran_pos, k=1, workers=-1)
nbr_count_direct = np.bincount(idx, minlength=Ntrc)

delta_direct = Ran_Fac_test/(nbr_count_direct + 1e-15) - 1.0
y_direct = 1.0/(1. + delta_direct)


#Compare the mean nbr_count:
print('Class method - mean nbr_count:',(Ran_Fac_test/(1+delta_class)).mean())
print('Direct method - mean nbr_count', nbr_count_direct.mean())

#Compare percentiles
percentiles = np.arange(1, 100)
pct_class = np.percentile(y_class, percentiles)
pct_direct = np.percentile(y_direct, percentiles)
 
plt.figure(figsize=(7,5))
plt.plot(percentiles, pct_class, '-r', label='voronoi_web class')
plt.plot(percentiles, pct_direct, '--b', label='direct cKDTree')
plt.yscale('log')
plt.xlabel('percentile')
plt.ylabel(r'$y = V/\langle V\rangle$')
plt.legend()
plt.title(f'VVF cross-check, Ran_Fac={Ran_Fac_test}')
plt.tight_layout()
plt.savefig('vvf_crosscheck.png', dpi=150)
print("Saved vvf_crosscheck.png")
