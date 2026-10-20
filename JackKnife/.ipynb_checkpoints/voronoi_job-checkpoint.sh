#!/bin/bash
#PBS -l select=1:ncpus=8:mem=50gb
#PBS -l walltime=05:00:00
#PBS -o rsd_jackknife_out.log
#PBS -e rsd_jackknife_err.log
#PBS -N jackknife_computation

ulimit -s 10240
cd $PBS_O_WORKDIR
PYTHON_EXE="/mnt/home/project/csatsangi.chetan/.conda/envs/halo_env/bin/python"
$PYTHON_EXE test_compute.py
