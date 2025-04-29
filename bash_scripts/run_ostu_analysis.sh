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

save_dir='/projects/OLIVINE/data/output/PARTICLE/Otsu/oryx/olivine_all_36fps_0.7x'
sand='Olivine'
exp='olivine_all_36fps_otsu_0.7x_no_edges'
echo "save_dir: "$save_dir

source /projects/OLIVINE/environments/SAM-grain/bin/activate

python3 -m analyze_threshold_grain_capture --save_dir $save_dir --sand_name $sand --exp $exp
