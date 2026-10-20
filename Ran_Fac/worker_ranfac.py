import numpy as np
from scipy.spatial import cKDTree
from astropy.table import Table
from numpy.lib import recfunctions as rfn
import sys

#Load data:
centrals = Table.read('centrals_sim.fits').as_array()
satellites = Table.read('satellites_sim.fits').as_array()
centrals_trimmed = rfn.drop_fields(centrals, ('Nsat','rvir'), usemask=False)
all_galaxies = np.concatenate([centrals_trimmed, satellites])

gpos = np.zeros((all_galaxies.shape[0], 3), dtype='float')
gpos[:,0] = all_galaxies['x']
gpos[:,1] = all_galaxies['y']
gpos[:,2] = all_galaxies['z']

#Parameters:
Ntrc = gpos.shape[0]
Lbox = 200.0
seed = 10
percentiles = np.arange(1, 100)
gpos = gpos%Lbox

task_id = int(sys.argv[1])
ran_fac_targets = np.array([1e3, 3e3, 1e4, 3e4, 1e5], dtype=int)
rf = ran_fac_targets[task_id]
print(f'starting task {task_id}, Ran_Fac = {rf}...')

#cKDTree Generation
print("Generating cKDTree")
tree = cKDTree(gpos, boxsize=Lbox)
rng = np.random.RandomState(10)

#Counting neighbours
print("Counting neighbours...")
Nran = rf * Ntrc
chunk_size = int(1e7) #Proceed chuink wise to optimize space
nbr_count = np.zeros(Ntrc, dtype=np.int64)
Nran_done = 0
while Nran_done < Nran:
    n = min(chunk_size, Nran - Nran_done) #at each iteration, choose the smaller number of points to generate
    ran = Lbox * rng.rand(n, 3)
    _, idx = tree.query(ran, k=1, workers=-1)
    nbr_count += np.bincount(idx, minlength=Ntrc)
    Nran_done += n
    del ran, idx

print("Computing vvf...")
delta = rf / (nbr_count + 1e-15) - 1.0
y = 1.0/(1.0 + delta)

print("Computing vvf_percentiles...")
pct = np.percentile(y, percentiles)

np.save(f'vvf_pct_ranfac_{rf}.npy', pct)
print(f"Ran_Fac={rf} done")