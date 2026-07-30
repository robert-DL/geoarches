#!/bin/sh
#SBATCH --job-name=backfill-missing-jan
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

. ~/.bashrc

module load python3/2025.01-gcc-13.3.0
source ~/repositories/geoenv/bin/activate

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "Running on ${SLURM_JOB_NUM_NODES} node(s) with ${SLURM_CPUS_PER_TASK} CPU(s)."
echo "Using ${SLURM_GPUS_PER_NODE} GPU(s)."

uid0="modelstore/AWM0-reduced_precip-relu"
uid1="modelstore/AWM1-reduced_precip-relu"
uid2="modelstore/AWM2-reduced_precip-relu"

srun --cpu-bind=none --mem-bind=none --gpus-per-node=4 --mem=0 --cpus-per-task=8 /work/bk1450/b383170/geoenv/bin/python \
    inference/backfill_missing_jan_predictions.py \
    --target-root /home/b/b383170/repositories/geoarches/geoarches/data/output/AWM-reduced_precip-relu \
    --uids ${uid0},${uid1},${uid2} \
    --steps 1,3,5,7 \
    --resplit-steps 1,3,5 \
    --skip-resplit \
    --python-bin /work/bk1450/b383170/geoenv/bin/python
