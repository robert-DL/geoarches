"""
Code to produce long-term rollouts with ArchesWeather and ArchesWeatherGen
"""
import torch
import hydra
import pathlib
import omegaconf

import xarray as xr
import numpy as np
import pandas as pd

from pprint import pprint


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
    def __init__(self, dataset: era5.Era5Forecast):
        self.dataset = dataset

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
        batch["state"] = torch.clone(output)

        # Forcings if applicable
        if "forcings" in batch.keys():
            new_forcings = self.dataset.load_forcings(datetime.to_numpy())
            new_forcings = new_forcings.unsqueeze(0).to(batch["state"]["level"].device)
            batch["forcings"] = new_forcings

        return batch

    def _to_nptime(self, datestring: str):
        return np.datetime64(datestring)
        
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
            dataset: era5.Era5Forecast
        ):

        super().__init__(dataset)
        self.module = module

    def rollout(self, start_datestring, end_datestring=None, freq=None, periods=None):
        dates, inital_batch = super().rollout(start_datestring, end_datestring, freq, periods)

        # Add batch dim
        inital_batch = collate_fn([inital_batch])

        # Move to correct device
        batch = {k: v.to(self.module.device) for k, v in inital_batch.items()}

        print(dates)
        for date in dates:
            datetime = pd.to_datetime(batch["timestamp"].detach().cpu()[0], unit="s")
            print(datetime)
            output = self.module(batch)

            batch = self.advance_batch(batch, output)

            print("######## DONE ######")
            
        

def test_rollouts():
    path = pathlib.Path(__file__).parent.parent
    print(path)

    def test_rollout_archesweather():
        overrides=[
            "dataloader=era5-aimip-overwrite", 
            "module=archesweather-aimip-overwrite", 
            "stats=aimip_stats"
        ]
        cfg = hydra.compose(config_name="config", overrides=overrides)
        omegaconf.OmegaConf.resolve(cfg)
        pprint(cfg)

        module = hydra.utils.instantiate(cfg.module.module, cfg.module, cfg.stats)
        dataset = hydra.utils.instantiate(cfg.dataloader.dataset, cfg.stats)

        aw_rollout = ArchesWeatherRollout(module, dataset)
        aw_rollout.rollout("2015-01-01T00", "2015-01-02T00", "1D")

    
    def test_rollout_archesweathergen():
        pass

    with hydra.initialize(config_path="../configs"):
        test_rollout_archesweather()


def test_checkpoints():
    path = "../modelstore/aimip-archesweather-m-seed0"
    module, cfg = load_module(path, return_cfg=True)

    dataset = era5.Era5Forecast(cfg.stats.module, cfg.dataloader.dataset)


            


test_checkpoints()