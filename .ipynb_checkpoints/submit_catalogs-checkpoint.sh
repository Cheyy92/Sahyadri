#!/bin/bash
#PBS -N gen_cosmo_mocks
#PBS -l nodes=1:ppn=4
#PBS -l walltime=02:00:00
#PBS -J 0-3
#PBS -o logs/gen_mock_$PBS_ARRAY_INDEX.out
#PBS -e logs/gen_mock_$PBS_ARRAY_INDEX.err

cd $PBS_O_WORKDIR

# Ensure log directory exists
mkdir -p logs

# Array of all 13 cosmological variation directories
VARIATIONS=(
    "var_Ok_m2048"
    "var_Ok_p2048"
    "var_w_m2048"
    "var_w_p2048"
)

# Pick the variation corresponding to the current job array index
VAR_NAME=${VARIATIONS[$PBS_ARRAY_INDEX]}

echo "Running catalog generation for: ${VAR_NAME}"
python "gen_var_cosm.py" "${VAR_NAME}"
echo "Completed ${VAR_NAME}"