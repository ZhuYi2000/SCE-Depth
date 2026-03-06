#!/bin/bash

module load apptainer/1.3.1
cd "$(dirname "$0")"
echo "Starting job"

# Set a unique port per job.
export MASTER_PORT=$((( RANDOM % 600 ) + 29400 ))

# SCE-Depth (HEAL-SWIN + SGFE + SGL)
python3 run.py --env singularity train_isaac --config_path=heal_swin/run_configs/depth_estimation/depth_swin_isaac_train_run_config.py

# HEAL-SWIN baseline
# python3 run.py --env singularity train_isaac --config_path=heal_swin/run_configs/depth_estimation/depth_swin_hp_train_run_config.py

# SWIN baseline
# python3 run.py --env singularity train_isaac --config_path=heal_swin/run_configs/depth_estimation/depth_swin_train_run_config.py
