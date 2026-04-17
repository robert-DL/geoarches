#!/bin/sh
#SBATCH --job-name=encode_dataset-hour12
#SBATCH --account=bk1450
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --exclusive
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
##SBATCH --dependency=afterany:23120704

. ~/.bashrc

module load python3/2025.01-gcc-13.3.0

source ~/repositories/geoenv/bin/activate # activate pip environment

export HYDRA_FULL_ERROR=1

export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "Running on ${SLURM_JOB_NUM_NODES} nodes with ${SLURM_CPUS_PER_TASK} CPUs per task."
echo "Using ${SLURM_GPUS_PER_NODE} GPUs."

uid0="modelstore/AWM0-aimip_default-unforced"
uid1="modelstore/AWM1-aimip_default-unforced"
uid2="modelstore/AWM2-aimip_default-unforced"
uid3="modelstore/AWM11-aimip_default-unforced"

srun --cpu-bind=none --mem-bind=none --gpus-per-node=4 --mem=0 --cpus-per-task=8 python3 inference/encode_dataset.py \
    --target-path="data/output/preds_awm-unforced-0x1x11x42" \
    --uids=${uid0},${uid1},${uid2},${uid3} \
    --encode-hours=12 \
    --forecast-steps=1,3,5,7