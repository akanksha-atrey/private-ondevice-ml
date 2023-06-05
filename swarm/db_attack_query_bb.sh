#!/bin/bash
#
#SBATCH --job-name=attack
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
echo "Running decision boundary attack..."

dataname=$1
modeltype=$2
noisebounds=$3

python3 -m src.attack.db_attack -data_name $dataname -model_type $modeltype -noise_bounds $noisebounds -exp_num_query_bb true 

wait

echo "Done running attack!"
exit
