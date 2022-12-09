#!/bin/bash

#SBATCH --job-name=Radiomics   # name of the job
#SBATCH --partition=casus    # partition to be used (defq, gpu or intel)
#SBATCH --time=19:00:00       # walltime (up to 96 hours)
#SBATCH --mem 70000
#SBATCH -v gpu
##SBATCH -C v100-32g
#SBATCH --gres=gpu:1
#SBATCH --nodes=1          # number of nodes
#SBATCH --ntasks=1     # number of tasks (i.e. parallel processes) to be started
#SBATCH --output=Radiomics%j.out      # nom du fichier de sortie
#SBATCH --error=Radiomics%j.out       # nom du fichier d'erreur (ici commun avec la sortie)

module purge

module load gcc
module load python/3.9.6
. ./CNNtorch/bin/activate
#Run this job with
# sbatch run.sh


# Execute your code
python /bigdata/casus/optima/scripts/CNN/Kaggle_Brain_MRI/Radiomics.py

