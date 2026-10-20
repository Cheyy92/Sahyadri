#!/bin/bash
#PBS -J 0-4
#PBS -l select=1:ncpus=20:mem=32gb
#PBS -l walltime=08:00:00
#PBS -N vvf_ranfac
#PBS -o ranfac_^array_index^_out.log
#PBS -e ranfac_^array_index^_out.log

cd $PBS_O_WORKDIR
PYTHON_EXE="/mnt/home/project/csatsangi.chetan/.conda/envs/halo_env/bin/python"
$PYTHON_EXE worker_ranfac.py $PBS_ARRAY_INDEX
