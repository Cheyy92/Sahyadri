import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from astropy.table import Table
from numpy.lib import recfunctions as rfn
import time

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

tree = cKDTree(gpos, boxsize=Lbox)
n_test = int(1e7)
ran = Lbox * np.random.rand(n_test, 3)

t0 = time.time()
_, idx = tree.query(ran, k=1, workers=-1)
dt = time.time() - t0
print(f"{n_test} queries in {dt:.2f}s -> {n_test/dt:.2e} queries/sec")