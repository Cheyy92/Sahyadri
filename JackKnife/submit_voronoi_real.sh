#!/bin/bash
#PBS -l select=1:ncpus=4:mem=128gb
#PBS -l walltime=03:00:00
#PBS -o voronoi_job_real.out
#PBS -e voronoi_job_real.err
#PBS -N jackknife_computation_real

ulimit -s 10240
cd $PBS_O_WORKDIR
PYTHON_EXE="/mnt/home/project/csatsangi.chetan/.conda/envs/halo_env/bin/python"
$PYTHON_EXE compute_voronoi_jack_real.py 64 > log_64.txt 2>&1 &
$PYTHON_EXE compute_voronoi_jack_real.py 144 > log_144.txt 2>&1 &
$PYTHON_EXE compute_voronoi_jack_real.py 100 > log_100.txt 2>&1 &

wait
