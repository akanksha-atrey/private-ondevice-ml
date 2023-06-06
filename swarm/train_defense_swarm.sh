#!/bin/bash
#
#SBATCH --job-name=train_ae
#SBATCH --output=./models/output/res_%j.txt             # output file
#SBATCH -e ./models/output/res_%j.err                   # File to which STDERR will be written
#SBATCH --partition=longq                              	# Partition to submit to
#
#SBATCH --ntasks=4
#SBATCH --time=10-01:00:00
#SBATCH --mem-per-cpu=32000                              # Memory in MB per cpu allocated

set -x

source /home/aatrey/mypython/bin/activate

echo "SLURM_JOBID: " $SLURM_JOBID
echo "Starting to train autoencoder based model..."

dataname=$1
modeltype=$2

python3 -m src.defense.defense_training -data_name $dataname -model_type $modeltype

echo "Done training the models!"
exit
