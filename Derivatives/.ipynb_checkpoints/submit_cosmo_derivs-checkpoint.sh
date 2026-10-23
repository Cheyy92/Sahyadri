#!/bin/bash
#PBS -N vvf_deriv_cosmo
#PBS -l select=1:ncpus=5:mem=32gb
#PBS -l walltime=02:00:00
#PBS -J 0-3

ulimit -s 10240
cd $PBS_O_WORKDIR

# Ensure log directory exists
mkdir -p logs

# 1. Define the parameters in a bash array (0-indexed)
PARAMS=("Om" "ns" "h" "As")

# 2. Extract the parameter for this specific array index
CURRENT_PARAM=${PARAMS[$PBS_ARRAY_INDEX]}

# 3. Redirect stdout and stderr dynamically to cleanly named files
exec > logs/deriv_cosmo_${CURRENT_PARAM}.out 2> logs/deriv_cosmo_${CURRENT_PARAM}.err

echo "Starting array job $PBS_ARRAY_INDEX"
echo "Computing VVF derivative for parameter: ${CURRENT_PARAM}"

PYTHON_EXE="/mnt/home/project/csatsangi.chetan/.conda/envs/halo_env/bin/python"

# 4. Execute the Python script
# Arguments: <param_to_vary> <rsd_boolean>
$PYTHON_EXE vvf_derivative_cosm.py $CURRENT_PARAM True

echo "Array job $PBS_ARRAY_INDEX completed."