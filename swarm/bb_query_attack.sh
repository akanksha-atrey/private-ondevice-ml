#!/bin/bash
#
#SBATCH --job-name=bb_query
#SBATCH --output=./results/output/res_%j.txt             # output file
#SBATCH -e ./results/output/res_%j.err                   # File to which STDERR will be written
#SBATCH --partition=longq                              	   # Partition to submit to
#
#SBATCH --ntasks=1
#SBATCH --time=10-01:00:00
#SBATCH --mem-per-cpu=30000                                # Memory in MB per cpu allocated

set -x

source /home/aatrey/mypython/bin/activate

echo "SLURM_JOBID: " $SLURM_JOBID
echo "Starting to run bb query attack..."

dataname=$1
modeltype=$2

python3 -m src.attack.class_attack -data_name $dataname -model_type $modeltype -bb_query_attack true

wait

echo "Done running attack!"
exit
