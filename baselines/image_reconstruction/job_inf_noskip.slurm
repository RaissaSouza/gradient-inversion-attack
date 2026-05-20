#! /bin/bash
# ====================================
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --time=07:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-v100
# ====================================

#SBATCH --job-name=unet_noskip_inference_harm
#SBATCH --output=unet_noskip_inference_harm.out

source /home/raissa.souzadeandrad/software/init-conda
conda activate /home/raissa.souzadeandrad/miniconda3/envs/inversion

python -u unet_noskip_inference.py

