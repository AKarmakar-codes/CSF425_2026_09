#!/usr/bin/env bash
#SBATCH --job-name=pcb_dl_train
#SBATCH --output=pcb_dl_%j.log
#SBATCH --error=pcb_dl_%j.err
#SBATCH --partition=gpu_rtx_pro_6000_6_csis_hyd,gpu_v100_2,gpu_a100_8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

# ==============================================================================
# CS F425 Deep Learning - HPC SLURM Batch Execution Script
# Submit to queue with: sbatch scripts/submit_hpc.sh
# ==============================================================================

echo "=== Job Started on $(hostname) at $(date) ==="
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# 1. Load system modules if required by your HPC
# module load anaconda3
# module load cuda/12.1

# 2. Activate Conda Environment
source ~/.bashrc
conda activate pcb_dl || source activate pcb_dl

# 3. Verify PyTorch sees the GPU
python -c "import torch; print('Using GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU FOUND!')"

# 4. Run all experiments sequentially
# This runs all 3 classification trials + all 3 segmentation trials,
# freezes validation thresholds, and saves best checkpoints.
python train_experiments.py --task all

echo "=== Job Finished at $(date) ==="

