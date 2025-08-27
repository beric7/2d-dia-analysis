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
img_dir='/path/to/directory/'
save_dir='/path/to/directory/'
sand='Venice_Beach'
exp='venice_beach_long1'
echo "Pixel Ratio: "$pixel_2_mm
echo "save_dir: "$save_dir

source /path/to/environment/bin/activate

python3 -m process_threshold_grain_capture --img_dir $img_dir --save_dir $save_dir --pixel2mm $pixel_2_mm --sand_name $sand --exp $exp
