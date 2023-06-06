#!/bin/bash
#
#SBATCH --job-name=detector
#SBATCH --output=./results/output/res_%j.txt             # output file
#SBATCH -e ./results/output/res_%j.err                   # File to which STDERR will be written
#SBATCH --partition=longq                              	   # Partition to submit to
#
#SBATCH --ntasks=4
#SBATCH --time=10-01:00:00
#SBATCH --mem-per-cpu=8000                                # Memory in MB per cpu allocated

set -x

source /home/aatrey/mypython/bin/activate

echo "SLURM_JOBID: " $SLURM_JOBID
echo "Running defense detector..."

dataname=$1
modeltype=$2
noisebounds=$3
numqueries=$4

python3 -m src.defense.defense_detector -data_name $dataname -model_type $modeltype \
                                        -noise_bounds $noisebounds -num_queries $numqueries

wait

echo "Done running defense!"
exit
