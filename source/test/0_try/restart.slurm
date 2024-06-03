#!/bin/bash
#SBATCH -J cp2k_u28
#SBATCH -p cnall
#SBATCH -N 1
#SBATCH -o %j.o
#SBATCH -e %j.e
#SBATCH --no-requeue
#SBATCH --ntasks-per-node=56

module load compilers/gcc/v9.3.0
source /apps/soft/cp2k/cp2k-9.1/tools/toolchain/install/setup

mpirun -np 56 /apps/soft/cp2k/cp2k-9.1/exe/local/cp2k.popt U28-1.restart > cp2k_2.out


