#!/bin/bash
#PBS -l select=1:ncpus=16:mem=32gb
#PBS -l walltime=05:00:00
#PBS -o vvf_conv_out.log
#PBS -e vvf_conv_err.log
#PBS -N vvf_convergence

ulimit -s 10240
cd $PBS_O_WORKDIR
PYTHON_EXE="/mnt/home/project/csatsangi.chetan/.conda/envs/halo_env/bin/python"
$PYTHON_EXE vvf_convergence.py
