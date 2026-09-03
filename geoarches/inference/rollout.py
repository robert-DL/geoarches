"""
Code to produce long-term rollouts with ArchesWeather and ArchesWeatherGen
"""
import os
import torch
import hydra

import xarray as xr
import numpy as np
import pandas as pd

from typing import Union, List

from geoarches.main_hydra import collate_fn
from geoarches.dataloaders import nan_util, era5
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

        os.makedirs(output_path, exist_ok=True)

    def apply_nan_mask(self, x, fill_nans=False):

        if fill_nans:
            x = nan_util.post_norm_interpolate_nans(
                x, "reset_nans"
            )
        else:
            x = nan_util.post_norm_interpolate_nans(
                    x, self.dataset.interpolate_input
            )
        return x

    def advance_batch(self, batch, output):
        datetime = pd.to_datetime(batch["timestamp"].detach().cpu()[0], unit="s")

        # Update time
        batch["hour_of_day"] = torch.tensor([datetime.hour]).to(batch["state"]["level"].device)
        batch["day_of_month"] = torch.tensor([datetime.day]).to(batch["state"]["level"].device)
        batch["day_of_year"] = torch.tensor([datetime.dayofyear]).to(batch["state"]["level"].device)
        batch["month"] = torch.tensor([datetime.month]).to(batch["state"]["level"].device)
        batch["timestamp"] = torch.tensor(
            [batch["timestamp"].detach().cpu().item() + 3600 * batch["lead_time_hours"].cpu().item()]
        )

        # Update conditioning
        batch["prev_state"] = torch.clone(batch["state"])
        batch["state"] = self.apply_nan_mask(torch.clone(output), False)

        # Forcings if applicable
        if "forcings" in batch.keys():
            new_forcings = self.dataset.load_forcings(datetime.to_numpy())
            new_forcings = self.apply_nan_mask(new_forcings, False)
            new_forcings = new_forcings.unsqueeze(0).to(batch["state"]["level"].device)
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

class ArchesWeatherRollout(Rollout):
    def __init__(
            self, 
            module: Union[AvgModule, ForecastModuleWithCond], 
            dataset: era5.Era5Forecast,
            output_path
        ):

        super().__init__(dataset, output_path=output_path)
        self.module = module

    @torch.no_grad()
    def rollout(self, start_datestring, end_datestring=None, freq=None, periods=None):
        dates, inital_batch = super().rollout(start_datestring, end_datestring, freq, periods)

        # Add batch dim
        inital_batch = collate_fn([inital_batch])

        # Move to correct device
        batch = {k: v.to(self.module.device) for k, v in inital_batch.items()}

        preds = []
        for i, date in enumerate(dates):
            datetime = pd.to_datetime(batch["timestamp"].detach().cpu()[0], unit="s")
            current_year = datetime.year
            if i > 0:
                timestamp = batch["timestamp"].detach().cpu()[0] + 3600 * self.lt
                next_year = pd.to_datetime(timestamp, unit="s").year
                if next_year > current_year:
                    # Year change
                    print(f"Dumping data for year {current_year}")
                    xrdataset = xr.concat(preds, dim="time")
                    xrdataset.to_netcdf(self.output_path  + f"/output_{current_year}.nc")
                    preds = []
                
            output = self.module(batch)
            batch = self.advance_batch(batch, output)

            preds.append(
                self.dataset.convert_to_xarray(
                    self.apply_nan_mask(self.dataset.denormalize(output), fill_nans=True), 
                    timestamp=batch["timestamp"].detach().cpu()
                )
            )

        if len(preds) > 0:
            print(f"Dumping data for year {current_year}")
            xrdataset = xr.concat(preds, dim="time")
            xrdataset.to_netcdf(self.output_path + f"/output_{current_year}.nc")
            preds = []

class ArchesWeatherGenRollout(ArchesWeatherRollout):
    def __init__(
            self, module: DiffusionModule, 
            dataset, output_path, member_idx=0,
            scale_input_noise=1.0, num_steps=25):
        super().__init__(module, dataset, output_path)

        self.scale_input_noise = scale_input_noise
        self.member_idx = member_idx
        self.num_steps = num_steps

    def rollout(self, start_datestring, end_datestring=None, freq=None, periods=None):
        dates, inital_batch = super().rollout(start_datestring, end_datestring, freq, periods)

        # Add batch dim
        inital_batch = collate_fn([inital_batch])

        # Move to correct device
        batch = {k: v.to(self.module.device) for k, v in inital_batch.items()}

        print(dates)
        preds = []
        for i, date in enumerate(dates):
            datetime = pd.to_datetime(batch["timestamp"].detach().cpu()[0], unit="s")
            current_year = datetime.year
            if i > 0:
                timestamp = batch["timestamp"].detach().cpu()[0] + 3600 * self.lt
                next_year = pd.to_datetime(timestamp, unit="s").year
                if next_year > current_year:
                    # Year change
                    print(f"Dumping data for year {current_year}")
                    xrdataset = xr.concat(preds, dim="time")
                    xrdataset.to_netcdf(self.output_path  + f"/output_{current_year}.nc")
                    preds = []
                
            output = self.module.sample(
                batch, 
                seed=self.member_idx,
                scale_input_noise=self.scale_input_noise,
                num_steps=self.num_steps
            )

            batch = self.advance_batch(batch, output)

            preds.append(
                self.dataset.convert_to_xarray(
                    self.apply_nan_mask(self.dataset.denormalize(output), fill_nans=True), 
                    timestamp=batch["timestamp"].detach().cpu()
                )
            )

        if len(preds) > 0:
            print(f"Dumping data for year {current_year}")
            xrdataset = xr.concat(preds, dim="time")
            xrdataset.to_netcdf(self.output_path + f"/output_{current_year}.nc")
            preds = []


@hydra.main(config_path="../configs", config_name="rollout")
def main(cfg):

    # Load model(s)
    module, mod_cfg = load_module(cfg.modelpaths, return_config=True)

    dataset = hydra.utils.instantiate(
        mod_cfg.dataloader.dataset, mod_cfg.stats, domain=cfg.rollout_domain
    )

    # Load rollout module
    rollout_op = hydra.utils.instantiate(cfg.rollout, module, dataset)

    rollout_op.rollout(**cfg.rollout_kwargs)
    

if __name__ == "__main__":
    main()