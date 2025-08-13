#!/bin/bash

#SBATCH --job-name=particle_analysis
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --output %u-%x-job%j.out
#SBATCH --export=ALL
#SBATCH --mem=20GB

# Print out Job Details
echo "Job ID: "$SLURM_JOB_ID
echo "Job Account: "$SLURM_JOB_ACCOUNT
echo "Hosts: "$SLURM_NODELIST
echo "------------"

exp='Olivine_sieve-140_36fps_amodal'
pixel_2_mm=0.00493177
save_dir='/path/to/directory/model_olivine/Olivine_sieve-140_36fps_amodal'
pred_dir='/path/to/directory/model_olivine/Olivine_sieve-140_36fps_amodal/Model_olivine'
sand='Olivine'
echo "Pixel Ratio: "$pixel_2_mm
echo "save_dir: "$save_dir

source /path/to/environment/bin/activate

python3 -m analyze_amodal_grain_capture --save_dir $save_dir --pred_dir $pred_dir --pixel2mm $pixel_2_mm --sand_name $sand --exp $exp

