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

pixel_2_mm=0.00493177
img_dir='/projects/OLIVINE/data/input_device/Oryx/oryx/venice/venice_all_long1_144fps'
save_dir='/projects/OLIVINE/data/output/PARTICLE/Otsu/oryx/venice_long1/'
sand='Venice_Beach'
exp='venice_beach_long1'
echo "Pixel Ratio: "$pixel_2_mm
echo "save_dir: "$save_dir

$(poetry env activate)

python3 -m analyze_threshold_grain_capture --img_dir $img_dir --save_dir $save_dir --pixel2mm $pixel_2_mm --sand_name $sand --exp $exp
