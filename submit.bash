#!/bin/bash
#SBATCH -J MonteCarlo
#SBATCH -A cs475-575
#SBATCH -p class
#SBATCH --gres=gpu:1
#SBATCH -o montecarlo.out
#SBATCH -e montecarlo.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hicksche@oregonstate.edu
for t in 16 32 64 128
do
    /usr/local/apps/cuda/cuda-10.1/bin/nvcc -DBLOCKSIZE=$t -o montecarlo montecarlo.cu
    ./montecarlo
done