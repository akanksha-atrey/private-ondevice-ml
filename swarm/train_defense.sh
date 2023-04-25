#!/bin/bash
#
#SBATCH --job-name=train_ae
#SBATCH --output=./models/output/res_%j.txt # output file
#SBATCH -e ./models/output/res_%j.err       # File to which STDERR will be written
#SBATCH --partition=gpu                     # Partition to submit to
#
#SBATCH -G 1								# Number of GPUs
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=32000                 # Memory in MB per cpu allocated
#SBATCH --time=10:00:00

set -x

module load python/3.9.1

echo "SLURM_JOBID: " $SLURM_JOBID
echo "Starting to train autoencoder based model..."

modeltype=$1

python3 -m src.defense_training -model_type $modeltype

echo "Done training the models!"

wait

exit
