#!/bin/bash

# Submit the job with a specific name
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

# pixel_2_mm=0.0057537
# pixel_2_mm=0.00493177
pixel_2_mm=0.00493177
img_dir='/projects/OLIVINE/data/synthetic_data/particle_test_B_500/particle_test_B_500_images'
save_dir='/projects/OLIVINE/data/output/PARTICLE/synth/test_B_500_threshold'
sand='Venice_Beach'
exp='olivine_all_36fps_1'
echo "Pixel Ratio: "$pixel_2_mm
echo "save_dir: "$save_dir

source /projects/OLIVINE/environments/SAM-grain/bin/activate

python3 -m process_threshold_grain_capture --img_dir $img_dir --save_dir $save_dir --pixel2mm $pixel_2_mm --sand_name $sand --exp $exp
