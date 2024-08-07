#!/bin/bash

# Node count
#SBATCH -N 1

# CPU core count 
#SBATCH -c 8

# Total RAM requirement (24GB)
#SBATCH --mem=24G

# Wall time (e.g. one day)
#SBATCH --time=20-00:00:00

# Write STDOUT to this file
#SBATCH --output=regression_no_norm.stdout

# Job name
#SBATCH --job-name="colab_regression_no_norm"

# Set the cluster partition to gpu
#SBATCH --partition="gpu"

# Make sure to specify the use of a generic resource - i.e. a single gpu
#SBATCH --gres=gpu:1

# Put commands for executing job below this line
module load Miniconda3/4.12.0

nvidia-smi

cd /scratch/kmmg694/image_collaboratorium_2024
source activate myenv  # Activate the Conda environment
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pandas
pip install scikit-learn
pip install tensorboard
pip install nibabel
pip install efficientnet_pytorch
pip install tqdm
python train.py
