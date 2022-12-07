#!/bin/bash

#SBATCH --job-name=Radiomics   # name of the job
#SBATCH --partition=casus    # partition to be used (defq, gpu or intel)
#SBATCH --time=19:00:00       # walltime (up to 96 hours)
#SBATCH --mem 70000
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1          # number of nodes
#SBATCH --ntasks=1     # number of tasks (i.e. parallel processes) to be started

module load gcc
module load python/3.9.6
. ./CNNtorch/bin/activate
#Run this job with
# sbatch run.sh

#module purge

#See available module with: module avail
#module load pytorch-gpu/py3/1.7.1 #it's difficult to use nnunet with this module
#module load gcc/8.3.1 cuda/10.2 nccl/2.7.8-1-cuda cudnn/8.0.4.30-cuda-10.2 intel-mkl/2020.1 magma/2.5.3-cuda openmpi/4.0.2-cuda #python/3.8.8

#Activate venv with this installation and python 3.8.28
#pip install torch torchvision torchaudio


# Execute your code
python Kaggle_Brain_MRI/Radiomics.py

