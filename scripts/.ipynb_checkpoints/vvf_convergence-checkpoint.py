import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from astropy.table import Table
from numpy.lib import recfunctions as rfn

#Load data:
centrals = Table.read('centrals.fits').as_array()
satellites = Table.read('satellites.fits').as_array()
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

gpos = gpos%Lbox

#-----Debugging-----#
p_test = 50 #pick a percentile
ran_fac_targets = np.arange(5000, 50001, 5000)
chunk_size = int(1e7)

tree = cKDTree(gpos, boxsize = Lbox)
rng = np.random.RandomState(seed)

nbr_count = np.zeros(Ntrc, dtype=np.int64)
Nran_done = 0
y_p_vals=[]

Nran_targets = ran_fac_targets * Ntrc

for target_Nran, rf in zip(Nran_targets, ran_fac_targets):
    while Nran_done < target_Nran:
        n = min(chunk_size, target_Nran-Nran_done)
        ran = Lbox * rng.rand(n, 3)
        _, idx = tree.query(ran, k=1, workers=-1)
        nbr_count += np.bincount(idx, minlength=Ntrc)
        Nran_done += n
        del ran,idx

    delta = rf / (nbr_count + 1e-15) - 1.0
    y = 1.0/(1.0 + delta)
    y_p = np.percentile(y, p_test)
    y_p_vals.append(y_p)
    print(f"Ran_Fac={rf}, y_{p_test} = {y_p:.5f}")

y_p_vals = np.array(y_p_vals)

#Plot : Ran_Fac/Ran_Fac_max vs VVF for this percentile:
plt.plot(ran_fac_targets / ran_fac_targets.max(), y_p_vals, 'o-')
plt.xlabel(r'$Ran\_Fac / Ran\_Fac_{max}$')
plt.ylabel(f'$y_{{{p_test}}}$')
plt.title(f'Convergence check: percentile {p_test}')
plt.tight_layout()
plt.savefig('Convergence_test.png', dpi=150)

#-----Debugging end-----#

