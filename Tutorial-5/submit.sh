#!/bin/bash
#SBATCH --job-name=matrix-multiplication
#SBATCH --gres=gpu:1
#SBATCH --output=matrix-multiplication.out
#SBATCH --error=matrix-multiplication.err
#SBATCH --time=00:05:00

nsys profile --stats=true -o naive_profile ./matrix-multiplication 0
nsys profile --stats=true -o tiled_profile ./matrix-multiplication 1
