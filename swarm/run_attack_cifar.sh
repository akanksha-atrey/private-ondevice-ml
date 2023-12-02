#!/bin/bash
#
#SBATCH --job-name=cifar_attack
#SBATCH --output=./results/output/res_%j.txt             # output file
#SBATCH -e ./results/output/res_%j.err                   # File to which STDERR will be written
#SBATCH --partition=longq                              	# Partition to submit to
#
#SBATCH --ntasks=4
#SBATCH --time=10-01:00:00
#SBATCH --mem-per-cpu=128000                              # Memory in MB per cpu allocated

set -x

source ./venv/bin/activate

echo "SLURM_JOBID: " $SLURM_JOBID
echo "Starting to run attack on pretrained CIFAR model..."

python -m src.attack.cifar_pretrained

echo "Done running the attack"
exit
