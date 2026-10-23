from astropy.table import Table
from numpy.lib import recfunctions as rfn
from hod_utils import HODFits_BGS
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
    print('Initializing the parameter and percent variations')
    Lbox = 200
    param_to_vary = sys.argv[1]
    percent = float(sys.argv[2])
    rsd = sys.argv[3].lower() in ['true', '1', 't', 'y', 'yes']

    if percent == 2.5:
        head = '2_5'
    else:
        head = '5'

    if (rsd):
        suff = 'rsd'
    else:
        suff = 'real'

    # Ensure output directory exists to avoid FileNotFoundError
    os.makedirs(f'derivatives/{head}', exist_ok=True)

    print('loading delta...')
    hod_fits = HODFits_BGS()
    params = hod_fits.get_hod_params_dict(-18.0)
    if (param_to_vary in ["logM1", "logMcut"]):
        f = percent/100
        delta = f/np.log(10)
    else:
        delta = percent*params[param_to_vary]/100.0
    
    file_param = param_to_vary.replace('log', '') if param_to_vary.startswith('log') else param_to_vary
    # load the plus and minus variation data:
    print(f'loading galaxy positions for plus and minus variations by {percent}%')
    centrals_plus = Table.read(f'plus_{head}/centrals_{suff}_{file_param}.fits').as_array()
    centrals_minus = Table.read(f'minus_{head}/centrals_{suff}_{file_param}.fits').as_array()

    satellites_plus = Table.read(f'plus_{head}/satellites_{suff}_{file_param}.fits').as_array()
    satellites_minus = Table.read(f'minus_{head}/satellites_{suff}_{file_param}.fits').as_array()

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
    print(f"Computing VVF percentiles in parallel for {param_to_vary} +/- {percent}%...")
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
    out_table.write(f'derivatives/{head}/dVVF_d{param_to_vary}_{suff}.fits', overwrite=True)
    print("Derivatives saved successfully.")