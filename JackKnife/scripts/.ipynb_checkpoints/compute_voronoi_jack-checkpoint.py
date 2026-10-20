import numpy as np  #numpy: https://numpy.org/doc/stable/index.html
import importlib.util  #importlib: https://docs.python.org/3/library/importlib.html
from astropy.table import Table
import time
import voronoi
import jackknife
import sys

#Input
njn = sys.argv[1]

#Load data:
gpos_jn = Table.read(f'rsd_results/gpos_jackknife_rsd_{njn}.fits').as_array()

gpos = np.zeros((gpos_jn.shape[0], 4), dtype='float')
gpos[:,0] = gpos_jn['col0']
gpos[:,1] = gpos_jn['col1']
gpos[:,2] = gpos_jn['col2']
gpos[:,3] = gpos_jn['col3']


#Parameters:
Ntrc = gpos.shape[0]
Lbox = 200.0
seed = 10
percentiles = np.arange(1, 100)
gpos[:, :3] = gpos[:, :3]%Lbox

#initializations:
vvf_handler = voronoi.Voronoi(positions=gpos[:, :3], Lbox=Lbox, seed=seed)
V_trc = vvf_handler.voronoi_periodic_box(ran_fac = 10000,
                             chunk_size=int(1e7),
                             return_counts=False)

percentiles = [2.5,16.0,50.0,84.0,97.5] #Hardcoded percentiles from sahyadri sandbox
jk = jackknife.JackKnife(positions=gpos[:, :3], Lbox=Lbox)
res = jk.jackknife_percentile_stats(V_trc, percentiles = percentiles, jnreg = gpos[:, 3])

output_table = Table()
output_table['perc'] = res['percentiles']
output_table['vvf_mean'] = res['mean']
output_table['vvf_err'] = res['err']
output_table['vvf_all'] = res['full']
output_table['vvf_realizations'] = res['realizations']  # shape (n_percentiles, njn)
output_table.write(f'vvf_results_{njn}.fits', format='fits', overwrite=True)

del V_trc
