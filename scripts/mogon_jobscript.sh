#!/bin/bash

#SBATCH -M mogonki
#SBATCH -J Cuckoo-Filter    # Job name
#SBATCH -o \%x_\%j.out      # Specify stdout output file where \%j expands to jobID and \%x to JobName
#SBATCH -A ki-gpu4science   # Account name
#SBATCH -p a100ai           # Queue name
#SBATCH -n 1                # Number of tasks
#SBATCH -c 32               # Number of CPUs
#SBATCH --gres=gpu:8        # Total number of GPUs
#SBATCH --mem=512G          # Memory per node
#SBATCH -t 2880             # Time in minutes

set -e

# shellcheck disable=SC1091
source ./scripts/mogon_env.sh

srun ./scripts/run_multi_gpu_scaling.py
