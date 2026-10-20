#!/bin/bash
#PBS -l select=1:ncpus=5:mem=32gb
#PBS -l walltime=02:00:00
#PBS -o rsd_catalog_out.log
#PBS -e rsd_catalog_err.log
#PBS -N rsd_catalog_generation

ulimit -s 10240
cd $PBS_O_WORKDIR
PYTHON_EXE="/mnt/home/project/csatsangi.chetan/.conda/envs/halo_env/bin/python"
$PYTHON_EXE generate_catalog.py
