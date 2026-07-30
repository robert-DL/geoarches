#!/bin/sh
#SBATCH --job-name=resplit-step07
#SBATCH --account=bk1450
#SBATCH --qos=normal
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

. ~/.bashrc

module load python3/2025.01-gcc-13.3.0
source ~/repositories/geoenv/bin/activate

export HYDRA_FULL_ERROR=1

echo "Running on ${SLURM_JOB_NUM_NODES} node(s) with ${SLURM_CPUS_PER_TASK} CPU(s)."

TARGET_DIR="/home/b/b383170/repositories/geoarches/geoarches/data/output/AWM-reduced_precip-relu/step07"

srun --cpu-bind=none --mem-bind=none --cpus-per-task=8 /work/bk1450/b383170/geoenv/bin/python \
    inference/resplit_yearly_netcdf_by_time.py \
    --input-dir "${TARGET_DIR}" \
    --delete-old
