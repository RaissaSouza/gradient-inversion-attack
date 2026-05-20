#! /bin/bash
# ====================================
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-v100
# ====================================

#SBATCH --job-name=unet_skip_training_seg
#SBATCH --output=unet_skip_training_seg.out

source /home/raissa.souzadeandrad/software/init-conda
conda activate /home/raissa.souzadeandrad/miniconda3/envs/inversion

python -u unet_reconstruction_model_training.py

