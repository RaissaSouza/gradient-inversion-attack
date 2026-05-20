#! /bin/bash
# ====================================
#SBATCH --job-name=grinv
#SBATCH --output=model-unet_bs-1_job-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-v100
#SBATCH --array=1-101%4
# ====================================

source /home/raissa.souzadeandrad/software/init-conda
conda activate /home/raissa.souzadeandrad/miniconda3/envs/inversion

# We batch five attacks per array job to get the runtime close to
# an hour. This is mainly a courtesy to not overtax the job scheduler.
readarray -t a < attack_targets.txt
for iter in $(seq 0 4);
do  
    idx=$(( 5 * $SLURM_ARRAY_TASK_ID + $iter ))
    subjid=${a[$idx]}
    image_path="/work/forkert_lab/harmonized_with_masked_include_adni/${subjid}.nii.gz"
    out_path="results/${subjid}"
    python attack_unet_parallellized.py -m unet_no_skip_model_2d_harm.pt -p norm_params.csv -i ${image_path} -o ${out_path}
done
