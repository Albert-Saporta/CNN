#!/bin/bash

#SBATCH --job-name=Radiomics   # name of the job
#SBATCH --partition=casus    # partition to be used (defq, gpu or intel)
#SBATCH --time=47:00:00       # walltime (up to 96 hours)
#SBATCH --mem 100000
#SBATCH --nodes=1          # number of nodes
##SBATCH -C v100-32g
#SBATCH --gres=gpu:2
#SBATCH --ntasks=2     # number of tasks (i.e. parallel processes) to be started
#SBATCH --output=radiomics%j.out      # nom du fichier de sortie
#SBATCH --error=radiomics%j.out       # nom du fichier d'erreur (ici commun avec la sortie)

module purge

module load gcc
module load python/3.9.6
. ./CNNtorch/bin/activate
#Run this job with
# sbatch run.sh


# Execute your code
python /bigdata/casus/optima/scripts/CNN/Kaggle_Brain_MRI/Radiomics.py

