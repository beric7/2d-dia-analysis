#!/bin/bash

declare -a jobs=(
  "south_high_part_3_144fps South_High model_B"
)

# Loop through each job in the list
for job in "${jobs[@]}"; do
  # Parse ID and INPUT_FOLDER from the job entry
  EXP=$(echo $job | awk '{print $1}')
  SAND=$(echo $job | awk '{print $2}')
  MODEL=$(echo $job | awk '{print $3}')

  # Submit the job with a specific name
  echo "Submitting job for EXP: $EXP, SAND: $SAND, MODEL: $MODEL"
  sbatch <<EOF
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
PIXEL_2_mm=0.00493177
SAVE_DIR="/projects/SSC-IMAGE-STITCHING/OLIVINE/data/output/PARTICLE/Amodal/oryx/$MODEL/$EXP/"
PRED_DIR="/projects/SSC-IMAGE-STITCHING/OLIVINE/data/output/PARTICLE/Amodal/oryx/$MODEL/$EXP/Model_B"
echo "Pixel Ratio: \$PIXEL_2_mm"
echo "pred_dir: \$PRED_DIR"
echo "save_dir: \$SAVE_DIR"
echo "exp name: \$EXP"

source /projects/OLIVINE/environments/SAM-grain/bin/activate

python3 -m analyze_amodal_grain_capture --save_dir "\$SAVE_DIR" --pred_dir "\$PRED_DIR" --pixel2mm "\$PIXEL_2_mm" --sand_name "\$SAND" --exp "\$EXP"

EOF
done
