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

# pixel_2_mm=0.0057537
# pixel_2_mm=0.00493177
exp='venice_beach_all_r1_36fps_amodal_model'
pixel_2_mm=0.00493177
save_dir='/projects/OLIVINE/data/output/PARTICLE/Amodal/oryx/model_B/venice_beach_all_07x_run1_36fps_0.2'
pred_dir='/projects/OLIVINE/data/output/PARTICLE/Amodal/oryx/model_B/venice_beach_all_07x_run1_36fps_0.2/venice_beach_all_07x_run1_36fps_0.2_model_B'
sand='Venice_Beach'
echo "Pixel Ratio: "$pixel_2_mm
echo "save_dir: "$save_dir

source /projects/OLIVINE/environments/SAM-grain/bin/activate

python3 -m analyze_amodal_grain_capture --save_dir $save_dir --pred_dir $pred_dir --pixel2mm $pixel_2_mm --sand_name $sand --exp $exp

