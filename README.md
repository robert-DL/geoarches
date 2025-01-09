# geoarches

geoarches is a machine learning package for training, running and evaluating ML models on weather and climate data, developped by Guillaume Couairon and Renu Singh in the ARCHES team at INRIA (Paris, France).

It can be used to run the ArchesWeather and ArchesWeatherGen weather models.

geoarches is based on pytorch, pytorch-lightning and hydra for configuration. After the package is installed, you can use its modules in python code, but you can also call the main training and evaluating scripts of geoarches. 


To develop your own models or make modifications to the existing ones, the intended usage is to write configurations files and pytorch lightning classes in your own working directory. Hydra will then discover your custom ``configs`` folder, and you can point to your custom classes from your custom config files.


## Installation

### Environment

Create an environment or activate the environment you are already using.

```sh
conda create --name weather python=3.10
conda activate weather
```

You can install the package in editable mode during development.
Editable mode allows you to make changes to the geoarches code locally, and these changes will automatically be reflected in your code that depends on it.

Move into this repo and type:
```sh
pip install -e .
pip install --no-dependencies tensordict
```
This also handles installing any dependencies.

### Useful directories

We recommend making the following symlinks in the codebase folder:
```sh
ln -s /path/to/data/ data
ln -s /path/to/models/ modelstore
ln -s /path/to/evaluation/ evalstore
ln -s /path/to/wandb/ wandblogs
```
Where `/path/to/models/` is where the trained models are stored, and `/path/to/evaluation/` is a folder used to store intermediate outputs from evaluating models. If you want to store models and data in your working directory, ou can also simply create regular folders.

### Downloading ArchesWeather and ArchesWeatherGen
Use following the script to download the 4 deterministic models (archesweather-m-...) and generative model (archesweathergen).

```sh
src="https://huggingface.co/gcouairon/ArchesWeather/resolve/main"
MODELS="archesweather-m-seed0 archesweather-m-seed1 archesweather-m-skip-seed0 archesweather-m-skip-seed1 archesweathergen"
for MOD in $MODELS; do
    mkdir -p modelstore/$MOD/checkpoints
    wget -O modelstore/$MOD/checkpoints/checkpoint.ckpt $src/${MOD}_checkpoint.ckpt
    wget -O modelstore/$MOD/config.yaml $src/${MOD}_config.yaml 
done
```
You can follow instructions in [`documentation/archesweather-tutorial.ipynb`](documentation/archesweather-tutorial.ipynb) to load the models and run inference with them. See [`documentation/archesweathergen_pipeline.md`](documentation/archesweathergen_pipeline.md) to run training.

## Running models with geoarches

The recommended way to use the package is to depend on the package inside your own working directory. Making edits directly in the geoarches package will make updates more difficult, but if you prefer this option, you can create a development branch so as to rebase it on future updates of geoarches.

### Using geoarches modules in python

After installing the geoarches package (see [Installation](#Installation)), you can use the geoarches tools directly by importing them from your directory, e.g.

```python
from geoarches.dataloaders.era5 import Era5Forecast
ds = Era5Foreacast(path='data/era5_240/full',
                   load_prev=True,
                   norm_scheme='pangu')
```


### Train models with CLI

To train model named `default_run`, you can run
```sh
python -m geoarches.main_hydra module=forecast-geoarchesweather dataloader=era5 \
++name=default_run
```
This will start a training for the deterministic model `ArchesWeather` on ERA5 data.

The model config will be saved to `modelstore/default_run/config.yaml` and the model checkpoints will be saved to `modelstore/default_run/checkpoints`.

Useful training options are 
```sh
python -m geoarches.main_hydra \
++log=True \ # log metrics on weights and biases
++seed=0 \ # set global seed
++cluster.gpus=4 \ # number of gpus used for distributed training
++batch_size=1 \ # batch size per gpu
++max_steps=300000 \ # maximum number of steps for training, but it's good to leave this at 300k for era5 trainings
++save_step_frequency=50000 \ # if you need to save checkpoints at a higher frequency
```

To run on a SLURM cluster, you can create a `configs/cluster` folder inside your working directory and put a ``custom_slurm.yaml`` configuration file in it with custom arguments. Then you call tell geoarches to use this configuration file with

```sh
python -m geoarches.submit --config-dir configs cluster=custom_slurm
```

### Evaluate models with CLI

To run evaluation of a model (e.g. ArchesWeather) on the test set (2020), you can run 
```sh 
MODEL=archesweather-m
python -m geoarches.main_hydra ++mode=test ++name=$MODEL
```
It will automatically load the config file in `modelstore/$MODEL` and load the latest checkpoint from ``modelstore/$MODEL/checkpoints``.
It will then run the metrics relevant for the loaded model (deterministic metrics for deterministic models and similarly for generative models)

Warning: if the provided model does not exist, it will not throw an error.

Useful options for testing:
```sh
python -m geoarches.main_hydra ++mode=test ++name=$MODEL \
++ckpt_filename_match=100000 \ # substring that should be present in checkpoint file name, e.g. here for loading the checkpoint at step 100000
++limit_test_batches=0.1 \ # run test on only a fraction of test set for debugging
++module.module.rollout_iterations=10 \ # autoregressive rollout horizon, in which case the line below is also needed
++dataloader.test_args.multistep=10 \ # allow the dataloader to load trajectories of size 10

++dataloader.test_args.

```

For testing the generative models, you can also use the following options:
```sh
++module.inference.num_steps=25 \ # num diffusion steps in generation
++module.inference.num_members=50 \ # num members in ensemble
++module.inference.rollout_iterations=10 \ # number of auto-regressive steps, 10 days by default.
```

## Implementing your own models


1) create a `configs` folder for you hydra configuration files. In this folder, you can put your own configs, e.g. by copying config files from geoarches and modifying them. Please note the config files should be put in the appropriate folder (`cluster`, `dataloader` or `module`) in you own `configs` folder.

2) Add new lightning modules or architectures in your working directory (we recommend putting lightning modules in a `lightning_modules` folder, and pytorch-only backbone architectures in a `backbones` folder). To tell hydra to use these modules, you can create a module config file `custom_forecast.yaml` in `configs/module` as following:
```yaml
module:
  _target_: lightning_modules.custom_module.CustomLightningModule
  ...

backbone:
  _target_: backbones.custom_backbone.CustomBackbone
  ...
```
You can of course mix and match you custom modules and backbones with the ones in geoarches.

3. Training models only requires one to tell hydra to use your `configs` folder with
```sh
python -m geoarches.main_hydra --config-dir configs
```

## Compute model outputs and metrics separately

You can compute model outputs and metrics separately. In that case, you first run evaluation as following:
```sh
python -m geoarches.main_hydra ++mode=test ++name=$MODEL \
++module.inference.save_test_outputs=False \
```

Then, to compute metrics, you can run `evaluation/eval_multistep.py` which reads in inference output from xarray files, computes specified metrics, and dumps metrics to `output_dir`. Example:

```sh
python -m geoarches.evaluation.eval_multistep \
    --pred_path evalstore/modelx_predictions/ \
    --output_dir evalstore/modelx_predictions/ \
    --groundtruth_path data/era5_240/full/  \
    --multistep 10 \
    --metrics era5_ensemble_metrics --num_workers 2
```

Before running, you need to make sure the metrics are registered in `evaluation/metric_registry.py` using register_metric(). You can find examples in the file. Example: 

    register_metric(
        "era5_ensemble_metrics",
        Era5EnsembleMetrics,
        save_memory=True,
    )

Metrics are registered with a name, class, and any arguments.

## Plot (WIP)

You can plot metrics for several models using the script `plot.py`. Just specify where the computed metrics are stored (either .nc or .pt files). Example:

python -m geoarches.evaluation.plot --output_dir plots/ \ --metric_paths /evalstore/modelx/...nc /evalstore/modely/...nc --model_names_for_legend ModelX ModelY \ 
--metrics rankhist --rankhist_prediction_timedeltas 1 7 \ --figsize 10 4 --vars Z500 Q700 T850 U850 V850

## Contributing to geoarches

If you want to contribute to geoarches, please see the [Contributing](CONTRIBUTING.md) section.

## External resources

Many thanks to the authors of WeatherLearn for adapting the Pangu-Weather pseudocode to pytorch. The code for our model is mostly based on their codebase.

[WeatherBench](https://sites.research.google/weatherbench/)

[WeatherLearn](https://github.com/lizhuoq/WeatherLearn/tree/master)