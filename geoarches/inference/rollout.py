"""
Code to produce long-term rollouts with ArchesWeather and ArchesWeatherGen
"""
import os
import torch
import hydra

import torch.multiprocessing as mp

import xarray as xr
import numpy as np
import pandas as pd

from typing import Union, List
from omegaconf import ListConfig

from geoarches.main_hydra import collate_fn
from geoarches.dataloaders import nan_util, era5, era5_constants
from geoarches.lightning_modules.base_module import load_module, AvgModule
from geoarches.lightning_modules.forecast import ForecastModuleWithCond
from geoarches.lightning_modules.diffusion import DiffusionModule

class Rollout:
    """
        This class is the base class to produce rollouts with the
        geoarches suite, either forced or unforced.
        Rollouts are started at a user specified date.
        The start_date is passed to the rollout and the (dervied) class
        implements the rollout methodology.

    """
    def __init__(self, dataset: era5.Era5Forecast, output_path):
        self.dataset = dataset
        self.lt = self.dataset.lead_time_hours
        self.output_path = output_path
        # True NaN mask (e.g. land for an ocean-only variable), captured once from the
        # raw initial batch before any interpolation. Model predictions never contain
        # real NaNs, so this persistent mask is what apply_nan_mask uses to reset/fill
        # those positions on every rollout step. Set in rollout() once the batch is built.
        self.nan_mask = None

        os.makedirs(output_path, exist_ok=True)

    def call_model(self, batch):
        """Runs one forecast step, dispatching to the right calling convention
        depending on whether the underlying module is a diffusion model."""
        if self.is_diffusion:
            return self.module.sample(
                batch,
                seed=self.member_idx,
                scale_input_noise=self.scale_input_noise,
                num_steps=self.num_steps,
            )
        return self.module(batch)

    def process_output(self, output):
        """Denormalizes model output, clips physically non-negative variables,
        and fills any remaining NaNs, in preparation for saving to disk."""
    
        output = self.dataset.denormalize(output)
        output = self.dataset.invert_log_transform(output)
        output = self.truncate_non_negative(output)
        output = self.apply_nan_mask(output, fill_nans=True, nan_mask=self.nan_mask)
        return output

    def apply_nan_mask(self, x, fill_nans=False, nan_mask=None):

        if fill_nans:
            x = nan_util.post_norm_interpolate_nans(
                x, "reset_nans", nan_mask=nan_mask
            )
        else:
            x = nan_util.post_norm_interpolate_nans(
                    x, self.dataset.interpolate_input, nan_mask=nan_mask
            )
        return x

    def truncate_non_negative(self, tdict):
        """
        Clip variables that are physically constrained to be zero or positive
        (e.g. specific humidity).

        Expects `tdict` to be in physical units (i.e. after denormalization),
        holding "surface" and/or "level" tensors with the variable dimension
        4th from last, matching the layout produced by convert_to_tensordict().
        """
        for key in ("surface", "level"):
            if key not in tdict:
                continue
            x = tdict[key]
            variables = self.dataset.variables[key]
            mask = torch.tensor(
                [v in era5_constants.non_negative_variables for v in variables],
                dtype=torch.bool,
                device=x.device,
            )
            if not mask.any():
                continue
            shape = [1] * x.dim()
            shape[-4] = mask.numel()
            tdict[key] = torch.where(mask.reshape(shape), x.clamp(min=0), x)
        return tdict

    def advance_batch(self, batch, output):
        device = batch["state"]["level"].device

        new_timestamp = (
            batch["timestamp"].detach().cpu()[0].item()
            + 3600 * batch["lead_time_hours"].cpu().item()
        )
        new_datetime = pd.to_datetime(new_timestamp, unit="s")

        # Update time to reflect the state produced by the model (output).
        batch["hour_of_day"] = torch.tensor([new_datetime.hour]).to(device)
        batch["day_of_month"] = torch.tensor([new_datetime.day]).to(device)
        batch["day_of_year"] = torch.tensor([new_datetime.dayofyear]).to(device)
        batch["month"] = torch.tensor([new_datetime.month]).to(device)
        batch["timestamp"] = torch.tensor([new_timestamp]).to(device)

        # Update conditioning
        batch["prev_state"] = torch.clone(batch["state"])

        # Get cloned version and apply proper chain to get values right
        cout = torch.clone(output)
        cout = self.dataset.invert_log_transform(self.dataset.denormalize(cout))
        cout = self.truncate_non_negative(cout)
        cout = self.dataset.apply_log_transform(cout)
        cout = self.dataset.normalize(cout)

        batch["state"] = self.apply_nan_mask(cout, False, nan_mask=self.nan_mask)


        # Forcings if applicable. load_forcings() internally looks up forcings for
        # (timestamp + lead_time_hours), so we must query it with the timestamp of
        # the state that was just produced (new_datetime) to get forcings for the
        # following step, not the ones already used as input this step.
        if "forcings" in batch.keys():
            new_forcings = self.dataset.load_forcings(new_datetime.to_numpy())
            new_forcings = self.dataset.normalize({"forcings": new_forcings})["forcings"]
            new_forcings = self.apply_nan_mask(new_forcings, False)
            new_forcings = new_forcings.unsqueeze(0).to(device)
            batch["forcings"] = new_forcings

        return batch

    def _to_nptime(self, datestring: str):
        return np.datetime64(datestring)

    @torch.no_grad()
    def rollout(self, start_datestring, end_datestring=None, freq=None, periods=None):
        """
            Method to invoke the rollout.
            args:
                start_datestring (str): Expects the start date as a string of
                the form yyyy-mm-ddThh. The str is converted into an npdatetime
                to obtain the index of the initial condition in the dataloader

        """
        assert isinstance(start_datestring, str), \
            "start date must be provided as a string"

        start_nptime = self._to_nptime(start_datestring)
        initial_batch = self.dataset.select_from_nptime(start_nptime)

        assert (end_datestring is not None and freq is not None) or \
            (freq is not None and periods is not None), \
            "Either end_datestring and freq must provided or freq and periods"

        if end_datestring is not None:
            dates = pd.date_range(
                start_nptime, self._to_nptime(end_datestring), freq=freq
                )
        elif periods is not None:
            dates = pd.date_range(
                start=start_nptime,
                periods=periods,
                freq=freq
            )

        return dates, initial_batch

class Forecast(Rollout):
    def __init__(
            self,
            module: Union[AvgModule, ForecastModuleWithCond, DiffusionModule],
            dataset: era5.Era5Forecast,
            forecast_steps: int, 
            output_path,
            member_idx=0,
            scale_input_noise=1.0,
            num_steps=25,
        ):

        super().__init__(dataset, output_path=output_path)
        self.module = module
        self.is_diffusion = isinstance(module, DiffusionModule)
        self.forecast_steps = forecast_steps
        self.member_idx = member_idx
        self.scale_input_noise = scale_input_noise
        self.num_steps = num_steps


    def dump_preds(self, preds, init_timestamp):

        xrdataset = xr.concat(preds, dim="time")

        xrdataset = xrdataset.rename({"time": "lead_time"})

        xrdataset = xrdataset.assign_coords(
            {"lead_time": self.lt * np.arange(1, xrdataset.lead_time.size+1)}
        )
        xrdataset = xrdataset.expand_dims(
            dim={"time":  pd.to_datetime(init_timestamp.unsqueeze(0), unit="s")},
            axis=0
        )

        init_datestring = pd.to_datetime(
            init_timestamp, unit="s").strftime("%Y-%m-%dT%H")
        xrdataset.to_netcdf(
            self.output_path + f"/output_{init_datestring}.nc")

    @torch.no_grad()
    def rollout(
        self, start_datestring, end_datestring=None, 
        freq=None, periods=None, days_of_month=None
    ):
        dates, _ = super().rollout(
            start_datestring, end_datestring, freq, periods)

        if days_of_month is None:
            dates = dates 
        else: 
            dates = dates[dates.day.isin(days_of_month)]


        preds = []
        preds_year = None
        for date in dates:
            batch = self.dataset.select_from_nptime(date.to_numpy())
            batch = collate_fn([batch])

            # Move to correct device
            batch = {k: v.to(self.module.device) for k, v in batch.items()}
            self.nan_mask = batch.get("nan_mask")

            init_timestamp = batch["timestamp"].detach().cpu()[0]
            print(
                pd.to_datetime(
                    init_timestamp, unit="s").strftime("%Y-%m-%dT%H")
            )
            print("------------------------------")
            for step in range(self.forecast_steps):
                if step % 5 == 0: print(f"[{step}/{self.forecast_steps}]")
                output = self.call_model(batch)
                batch = self.advance_batch(batch, output)

                output = self.process_output(output)
                preds.append(
                    self.dataset.convert_to_xarray(
                        output, 
                        timestamp=batch["timestamp"].detach().cpu()
                    )
                )

            # Dump and reset accumulated predictions whenever we cross a year boundary,
            # so that each output file only ever holds a single calendar year.
            self.dump_preds(
                preds, 
                init_timestamp,
            )

            preds = []
            
            print("------------------------------\n")


class ClimateRollout(Rollout):
    """
        Rollout class for both ArchesWeather (deterministic) and ArchesWeatherGen
        (diffusion) modules. The model calling convention differs between the two
        (a plain forward pass vs. iterative diffusion sampling), and is dispatched
        in call_model() based on the type of the passed module. Diffusion-only
        arguments (member_idx, scale_input_noise, num_steps) are ignored when
        module is not a DiffusionModule.
    """
    def __init__(
            self,
            module: Union[AvgModule, ForecastModuleWithCond, DiffusionModule],
            dataset: era5.Era5Forecast,
            output_path,
            member_idx=0,
            scale_input_noise=1.0,
            num_steps=25,
        ):

        super().__init__(dataset, output_path=output_path)
        self.module = module
        self.is_diffusion = isinstance(module, DiffusionModule)

        self.member_idx = member_idx
        self.scale_input_noise = scale_input_noise
        self.num_steps = num_steps

    def call_model(self, batch):
        """Runs one forecast step, dispatching to the right calling convention
        depending on whether the underlying module is a diffusion model."""
        if self.is_diffusion:
            return self.module.sample(
                batch,
                seed=self.member_idx,
                scale_input_noise=self.scale_input_noise,
                num_steps=self.num_steps,
            )
        return self.module(batch)



    def dump_preds(self, preds, year):
        print(f"Dumping data for year {year}")
        xrdataset = xr.concat(preds, dim="time")
        xrdataset.to_netcdf(self.output_path + f"/output_{year}.nc")

    @torch.no_grad()
    def rollout(self, start_datestring, end_datestring=None, freq=None, periods=None):
        dates, initial_batch = super().rollout(start_datestring, end_datestring, freq, periods)

        # Add batch dim
        initial_batch = collate_fn([initial_batch])

        # Move to correct device
        batch = {k: v.to(self.module.device) for k, v in initial_batch.items()}
        self.nan_mask = batch.get("nan_mask")

        preds = []
        preds_year = None
        for _ in dates:
            output = self.call_model(batch)
            batch = self.advance_batch(batch, output)

            output = self.process_output(output)
            timestamp = batch["timestamp"].detach().cpu()
            year = pd.to_datetime(timestamp[0].item(), unit="s").year

            # Dump and reset accumulated predictions whenever we cross a year boundary,
            # so that each output file only ever holds a single calendar year.
            if preds_year is not None and year != preds_year and len(preds) > 0:
                self.dump_preds(preds, preds_year)
                preds = []

            preds_year = year
            preds.append(
                self.dataset.convert_to_xarray(
                    self.apply_nan_mask(output, True, nan_mask=self.nan_mask), timestamp=timestamp
                )
            )

        if len(preds) > 0:
            self.dump_preds(preds, preds_year)



def _setup(cfg, modelpath, device="auto"):
    # Load model(s)
    module, mod_cfg = load_module(modelpath, device=device, return_config=True)

    dataset = hydra.utils.instantiate(
        mod_cfg.dataloader.dataset, mod_cfg.stats, domain=cfg.rollout_domain
    )

    return module, dataset


def _as_list(modelpaths):
    if isinstance(modelpaths, (list, ListConfig)):
        return list(modelpaths)
    return [modelpaths]


def multiprocess(rank, world_size, cfg):
    """Runs one rollout process pinned to a single GPU (`rank`).

    If `cfg.modelpaths` holds several paths, each process loads a different
    (independently trained) model, one per GPU. Otherwise, a single model is
    loaded on every process and each is assigned a different ensemble member
    (`member_idx=rank`), which only matters for a diffusion model.
    """
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    modelpaths = _as_list(cfg.modelpaths)
    is_ensemble = len(modelpaths) > 1
    modelpath = modelpaths[rank] if is_ensemble else modelpaths[0]

    module, dataset = _setup(cfg, modelpath, device=device)

    # Keep each process' outputs separate so parallel runs don't overwrite
    # each other: one output dir per model when running distinct models,
    # one per member when running the same (diffusion) model repeatedly.
    suffix = os.path.basename(str(modelpath).rstrip("/")) if is_ensemble else f"member_{rank}"
    output_path = os.path.join(
        os.path.dirname(cfg.rollout.output_path.rstrip("/")), suffix
    )

    rollout_op = hydra.utils.instantiate(
        cfg.rollout, module, dataset, output_path=output_path, member_idx=rank
    )
    rollout_op.rollout(**cfg.rollout_kwargs)


@hydra.main(config_path="../configs", config_name="rollout")
def main(cfg):
    ngpus = torch.cuda.device_count()
    modelpaths = _as_list(cfg.modelpaths)

    if len(modelpaths) > 1:
        # One distinct model per GPU.
        nprocs = len(modelpaths)
    elif ngpus > 1:
        # Single model: use every GPU on the node to run one ensemble
        # member each (relevant for diffusion models).
        nprocs = ngpus
    else:
        nprocs = 1

    if nprocs > 1:
        assert nprocs <= ngpus, (
            f"Requested {nprocs} process(es) but only {ngpus} GPU(s) are visible."
        )
        mp.spawn(multiprocess, args=(nprocs, cfg), nprocs=nprocs)
        return

    module, dataset = _setup(cfg, modelpaths[0])

    # Load rollout module
    rollout_op = hydra.utils.instantiate(cfg.rollout, module, dataset)
    rollout_op.rollout(**cfg.rollout_kwargs)
if __name__ == "__main__":
    main()
