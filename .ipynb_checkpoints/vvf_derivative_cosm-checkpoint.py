from astropy.table import Table
from numpy.lib import recfunctions as rfn
import sys
import numpy as np
import concurrent.futures
import os

sys.path.insert(0, "/mnt/home/project/csatsangi.chetan/projects/scripts")
from voronoi import Voronoi


def compute_vvf_percentiles(gpos, Lbox, ran_fac=10000):
    """
    Helper function to run the Voronoi calculation(CPU Heavy) and return percentiles.
    """
    vvf_helper = Voronoi(gpos, Lbox=Lbox)
    y = vvf_helper.voronoi_periodic_box(ran_fac=ran_fac)
    percentiles = [2.5, 16.0, 50.0, 84.0, 97.5]
    return np.percentile(y, percentiles)


if __name__ == '__main__':
    # initialize the parameter and percentage:
    print('Initializing cosmological parameters...')
    Lbox = 200
    param_to_vary = sys.argv[1] # 'Om', 'h', 'ns', or 'As'
    rsd = sys.argv[2].lower() in ['true', '1', 't', 'y', 'yes']
    
    suff = "rsd" if rsd else 'real'

    os.makedirs('derivatives/cosmo', exist_ok=True)

    # 1. Exact cosmological step sizes calculated from the Sahyadri table
    cosmo_deltas = {
        'Om': 0.01569,
        'ns': 0.048245,
        'h': 0.03368,
        'As': 2.0989e-10,
        'w': 0.5,
        'Ok': 0.05
    }

    if param_to_vary not in cosmo_deltas:
        raise ValueError(f"Delta for {param_to_vary} is missing from the cosmo_deltas dictionary.")
    
    delta = cosmo_deltas[param_to_vary]
    
    # 2. load the plus and minus variation data:
    print(f'Loading galaxy positions for var_{param_to_vary}_p/m2048')
    
    centrals_plus = Table.read(f'var_{param_to_vary}_p2048/centrals_{suff}.fits').as_array()
    centrals_minus = Table.read(f'var_{param_to_vary}_m2048/centrals_{suff}.fits').as_array()

    satellites_plus = Table.read(f'var_{param_to_vary}_p2048/satellites_{suff}.fits').as_array()
    satellites_minus = Table.read(f'var_{param_to_vary}_m2048/satellites_{suff}.fits').as_array()

    centrals_trimmed_plus = rfn.drop_fields(centrals_plus, ('Nsat', 'rvir'), usemask=False)
    all_galaxies_plus = np.concatenate([centrals_trimmed_plus, satellites_plus])

    gpos_plus = np.zeros((all_galaxies_plus.shape[0], 3), dtype='float')
    gpos_plus[:, 0] = all_galaxies_plus['x']
    gpos_plus[:, 1] = all_galaxies_plus['y']
    gpos_plus[:, 2] = all_galaxies_plus['z']

    centrals_trimmed_minus = rfn.drop_fields(centrals_minus, ('Nsat', 'rvir'), usemask=False)
    all_galaxies_minus = np.concatenate([centrals_trimmed_minus, satellites_minus])

    gpos_minus = np.zeros((all_galaxies_minus.shape[0], 3), dtype='float')
    gpos_minus[:, 0] = all_galaxies_minus['x']
    gpos_minus[:, 1] = all_galaxies_minus['y']
    gpos_minus[:, 2] = all_galaxies_minus['z']

    # Execute the plus and minus VVF calculations in parallel using 2 workers
    print(f"Computing VVF percentiles in parallel for {param_to_vary} ...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        future_plus = executor.submit(compute_vvf_percentiles, gpos_plus, Lbox)
        future_minus = executor.submit(compute_vvf_percentiles, gpos_minus, Lbox)

        # This will block until both background processes finish
        y_p_plus = future_plus.result()
        y_p_minus = future_minus.result()

    # Compute derivative of vvf using Central Difference:
    derivative = (y_p_plus - y_p_minus)/(2*delta)
    out_table = Table([derivative], names=[f'dVVF_d{param_to_vary}'])

    # Save output
    out_table.write(f'derivatives/cosmo/dVVF_d{param_to_vary}_{suff}.fits', overwrite=True)
    print("Derivatives saved successfully.")