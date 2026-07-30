#!/bin/sh
#SBATCH --job-name=AWM2-reduced_precip-relu
#SBATCH --account=bk1450
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --exclusive
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
##SBATCH --dependency=afterany:26219274

. ~/.bashrc

module load python3/2025.01-gcc-13.3.0

source ~/repositories/geoenv/bin/activate # activate pip environment

export HYDRA_FULL_ERROR=1

export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "Running on ${SLURM_JOB_NUM_NODES} nodes with ${SLURM_CPUS_PER_TASK} CPUs per task."
echo "Using ${SLURM_GPUS_PER_NODE} GPUs."

######### Regular Training without Multistep / 1.0 Degree, 3x3 kernel, aimip #########

### GENERIC ###
name=${SLURM_JOB_NAME}
echo "Experiment Name: ${name}"
log=True
seed=2
save_step_frequency=10000
cpus_per_task=8
max_steps=300000

### DATA ###
era5_path="data/era5_1x1_daily/full"
lead_time_hours=24
interpolate_input="zero_after_norm"
interpolate_target="none"
warning_on_nan=False
domain="daily_train"
val_domain="daily_val"
switch_recent_data_after_steps=300000
dataloader_dataset_forcings_path=Null #"data/era5_1x1/ERA5-1x1-monthly-mean-forcing-1978-2024_regridded.nc"
dataloader_dataset_forcings_stats_path=Null #"/home/b/b383170/repositories/geoarches/geoarches/data/era5_1x1/ERA5-0.25deg-monthly-mean-forcing-1978-2013_regridded_conservative_norm_stats.nc"
dataloader_dataset_forcing_vars=Null #["sea_surface_temperature","sea_ice_cover"]

### WANDB ###
wandb_mode="online" # "offline" or "online"
wandb_entity="deep-climate"
wandb_project="geoclimate"

### STATS ###
norm_file="daily_norm_stats_w_log.nc"
variables_surface=[10m_u_component_of_wind,10m_v_component_of_wind,2m_temperature,sea_surface_temperature,mean_sea_level_pressure,sea_ice_cover,total_precipitation]
loss_weight_per_variable_surface=[0.1,0.1,1.0,0.1,0.1,0.1,0.1]
latitude=181

### MODEL ###
patch_size=[2,3,3]
img_size=[13,181,360]
forcings_ch=0
forcings_embedding=Null
surface_ch=7  # length of variables_surface
emb_dim=192
out_emb_dim=384
backbone_window_size=[1,6,10]
backbone_tensor_size=[8,60,120]
backbone_emb_dim=192
backbone_num_heads=[6,12,12,6]
lr=3.0e-4
depth_multiplier=2
constant_mask_file="archesweather_constant_masks_1x1"
padding_mode="latlon"
cond_times=["day_of_year"]
apply_relu='{surface:[-1]}'

srun --cpu-bind=none --mem-bind=none --gpus-per-node=4 --mem=0 --cpus-per-task=8 python3 main_hydra.py \
    ++name=${name} \
    ++wandb_name=${name} \
    ++max_steps=${max_steps} \
    ++dataloader.dataset.path=${era5_path} \
    ++dataloader.dataset.lead_time_hours=${lead_time_hours} \
    ++seed=${seed} \
    ++log=${log} \
    ++entity=${wandb_entity} \
    ++project=${wandb_project} \
    ++save_step_frequency=${save_step_frequency} \
    ++cluster.cpus=${cpus_per_task} \
    ++cluster.wandb_mode=${wandb_mode} \
    ++dataloader.dataset.domain=${domain} \
    ++dataloader.validation_args.domain=${val_domain} \
    ++dataloader.dataset.switch_recent_data_after_steps=${switch_recent_data_after_steps} \
    ++dataloader.dataset.warning_on_nan=${warning_on_nan} \
    ++dataloader.dataset.interpolate_target=${interpolate_target} \
    ++dataloader.dataset.interpolate_input=${interpolate_input} \
    ++stats.module.norm_file=${norm_file} \
    ++stats.module.variables.surface=${variables_surface} \
    ++stats.compute_loss_coeffs_args.loss_weight_per_variable.surface=${loss_weight_per_variable_surface} \
    ++stats.compute_loss_coeffs_args.latitude=${latitude} \
    ++module.module.lr=${lr} \
    ++module.module.cond_times=${cond_times} \
    ++module.module.apply_relu=${apply_relu} \
    ++module.embedder.img_size=${img_size} \
    ++module.embedder.patch_size=${patch_size} \
    ++module.embedder.surface_ch=${surface_ch} \
    ++module.embedder.constant_mask_file=${constant_mask_file} \
    ++module.embedder.emb_dim=${emb_dim} \
    ++module.embedder.out_emb_dim=${out_emb_dim} \
    ++module.embedder.padding_mode=${padding_mode} \
    ++module.backbone.window_size=${backbone_window_size} \
    ++module.backbone.tensor_size=${backbone_tensor_size} \
    ++module.backbone.emb_dim=${backbone_emb_dim} \
    ++module.backbone.num_heads=${backbone_num_heads} \
    ++module.backbone.depth_multiplier=${depth_multiplier} \
    ++module.embedder.forcings_ch=${forcings_ch} \
    ++module.embedder.forcings_embedding=${forcings_embedding} \
    ++dataloader.dataset.forcings_path=${dataloader_dataset_forcings_path} \
    ++dataloader.dataset.forcing_vars=${dataloader_dataset_forcing_vars} \
    ++dataloader.dataset.forcings_stats_path=${dataloader_dataset_forcings_stats_path} 


