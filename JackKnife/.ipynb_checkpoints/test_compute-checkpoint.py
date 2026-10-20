import numpy as np  #numpy: https://numpy.org/doc/stable/index.html
import importlib.util  #importlib: https://docs.python.org/3/library/importlib.html
from astropy.table import Table
import time
import voronoi



#Load data:
gpos_jn = Table.read('gpos_jackknife_rsd.fits').as_array()

gpos = np.zeros((gpos_jn.shape[0], 4), dtype='float')
gpos[:,0] = gpos_jn['col0']
gpos[:,1] = gpos_jn['col1']
gpos[:,2] = gpos_jn['col2']
gpos[:,3] = gpos_jn['col3']

njn = len(np.unique(gpos_jn['col3']))


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
vvf_arr = np.zeros((len(percentiles),njn+4),dtype=float)
vvf_arr[:,0]=percentiles

#Building the output:
for i in range(0, njn):
    index_mask = (gpos[:, -1]!=i)
    vvf_arr[:,i+4] = np.percentile(V_trc[index_mask], percentiles)
    
vvf_arr[:,1] = np.mean(vvf_arr[:,4:], axis=1)
vvf_arr[:,2] = np.sqrt(njn-1)*np.std(vvf_arr[:,4:], axis=1)
vvf_arr[:,3]=np.percentile(V_trc,percentiles)
head = 'perc, vvf_mean, vvf_err, vvf_all, vvf_realizations'

output_table = Table()
output_table['perc'] = vvf_arr[:, 0]
output_table['vvf_mean'] = vvf_arr[:, 1]
output_table['vvf_err'] = vvf_arr[:, 2]
output_table['vvf_all'] = vvf_arr[:, 3]

# vvf_arr[:, 4:] grabs everything from column 4 to the end
output_table['vvf_realizations'] = vvf_arr[:, 4:]
output_table.write('vvf_results.fits', format='fits', overwrite=True)


del V_trc
