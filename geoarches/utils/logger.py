"""
Custom GeoarchesLogger that circumvents problems with wandb and online logging
when online logging is not available
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import argparse
import yaml
import logging

from mpl_toolkits.axes_grid1 import make_axes_locatable

import os
import csv
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.loggers.csv_logs import ExperimentWriter


############## PLOT SETTINGS ###################
mpl.rcParams["lines.linewidth"] = 2

base_colors = [
    [0,0,0],
    [0.902,0.624,0],
    [0.337,0.706,0.914],
    [0,0.620,0.451],
    [0.941,0.894,0.259],
    [0,0.447,0.698],
    [0.835,0.369,0],
    [0.8,0.475,0.655]
]

def lighten(color, factor=0.5):
    """Blends `color` toward white by `factor` (0 = unchanged, 1 = white)."""
    return [c + (1 - c) * factor for c in color]

# Base colors first, then a lighter tint of each, so cycling past the base palette
# (e.g. more than 8 lines/experiments) still gives visually distinct colors instead
# of repeating.
cycler = mpl.cycler(color=base_colors + [lighten(c) for c in base_colors])

mpl.rcParams['axes.prop_cycle'] = cycler

plt.rc('font', size=8)          # Controls default text sizes
plt.rc('axes', titlesize=10)     # Font size of the axes title
plt.rc('axes', labelsize=8)     # Font size of the x and y labels
plt.rc('xtick', labelsize=7)    # Font size of the tick labels
plt.rc('ytick', labelsize=7)    # Font size of the tick labels
plt.rc('legend', fontsize=8)    # Legend font size
plt.rc('figure', titlesize=12)   # Font size of the figure title

argparse.ArgumentParser()


############## Custom Logging ###################

class AppendExperimentWriter(ExperimentWriter):
    """ExperimentWriter that keeps metrics.csv across resumed training runs.

    Each resume constructs a fresh CSVLogger/ExperimentWriter, and the parent
    class's __init__ calls _check_log_dir_exists(), which deletes any existing
    metrics.csv so the run always starts from an empty file. Skip that so the
    file survives, and rely on ExperimentWriter.save() (unchanged), which
    already appends to metrics.csv when it exists and only rewrites the file
    when new metric columns show up (adding them to the header).
    """

    def __init__(self, log_dir: str) -> None:
        super().__init__(log_dir)
        # save() only rewrites the header to include columns not yet in
        # self.metrics_keys, which starts empty on this fresh instance. Seed
        # it with the existing file's header so columns logged before this
        # resume (e.g. val_* columns not re-logged until the next validation
        # epoch) aren't dropped when the header is next rewritten.
        if self._fs.isfile(self.metrics_file_path):
            with self._fs.open(self.metrics_file_path, "r", newline="") as file:
                reader = csv.DictReader(file)
                if reader.fieldnames:
                    self.metrics_keys = list(reader.fieldnames)

    def _check_log_dir_exists(self) -> None:
        pass


class AppendingCSVLogger(CSVLogger):
    @property
    def experiment(self) -> AppendExperimentWriter:
        if self._experiment is None:
            os.makedirs(self.log_dir, exist_ok=True)
            self._experiment = AppendExperimentWriter(self.log_dir)
        return self._experiment


############## SMOOTHING HELPER ###################
class RunningEMA:
    """
        Recursive Formulation of the Running EMA
        Y(i) = w * X(i) + (1-w) * Y(i-1)

    """
    def __init__(self, weight=0.8, window_size=10):
        self.weight = weight
        self._data = None
        self.window_size = window_size
        self.current_min = 0

    def _rema(self, i):
        if i == self.current_min:
            return self._data[i]
        else:
            self._data[i] = self.weight * self._data[i] \
                + (1-self.weight) * self._rema(i-1)

            return self._data[i]
            

    def compute_rma(self, array):
        self._data = np.copy(array)
        for k in range(1, (len(self._data) - self.window_size) // self.window_size + 1):
            if k * self.window_size > len(self._data):
                i = len(self._data)
            else:
                i = k * self.window_size

            _ = self._rema(i)
            self.current_min = k * self.window_size

        return self._data


############## PLOT LOG DATA ###################
class VisualizeLogs:
    logger_metrics = {
        "train_rmse_": "Train RMSE",
        "val_rmse_": "Validation RMSE"
    }

    logger_vars = {
        "SST": "Sea surface temperature",
        "TP": "Total precipitation",
        "T2m": "Temperature at surface",
        # to be continued
    }

    def __init__(
        self, 
        exp_dir, 
        modelnames, 
        target_logger_metrics: dict = {}, 
        output_dir=None, 
        apply_smoothing=False, 
        ema_weight=0.8,
        ema_window=10):
        super().__init__()

        paths = [exp_dir + f"/{name}" for name in modelnames]
        if output_dir is None:
            self.output_dir = paths[0]
        else:
            self.output_dir = output_dir

        self.data  = [self.average_non_unqiue_steps(pd.read_csv(f"{p}/metrics.csv")) for p in paths]
        
        self.hyparams = []
        self.labels = []
        for p in paths:
            with open(os.path.join(p,"hparams.yaml")) as stream:
                cfg = yaml.safe_load(stream)
                self.hyparams.append(cfg)
                self.labels.append(cfg["name"])
            
        self.logger_metrics = target_logger_metrics or self.logger_metrics

        # Smoothing params
        self.ema_weight = ema_weight
        self.apply_smoothing = apply_smoothing
        self.ema_window = ema_window

    def average_non_unqiue_steps(self, df):
        """
            The current logger appends steps that are already in the data
            when the run is resumed from an earlier checkpoint. We can account 
            for that by simply averaging the values out
        """

        return df.groupby("step", as_index=False).mean()

    def var_lineplot(self, ax, clmn, metric):
        for df, lbl in zip(self.data, self.labels):
            if clmn not in df.columns:
                logging.info(f"{clmn} not available for {lbl}, skipping")
                continue

            if "val" in clmn:
                x = "epoch"
            else:
                x = "step"

            # Metrics are logged on different rows (e.g. val metrics are only
            # populated on the row logged at the end of the epoch), so most
            # rows have NaN for this column. Drop them, otherwise no two
            # logged points are adjacent and matplotlib draws no line.
            df = df.dropna(subset=[clmn])
            data = self._smooth_data(df[clmn])
            df.loc[:, clmn] = data
            df.plot(
                x=x,
                y=clmn, 
                ax=ax, 
                grid=True, 
                logx=False, 
                logy=True, 
                xlabel=x, 
                ylabel=f"{self.logger_metrics[metric]}"
            )

        return ax

    def _smooth_data(self, x):
        if self.apply_smoothing:
            return RunningEMA(self.ema_weight, self.ema_window).compute_rma(x)
        else:
            return x

    def lineplot(self, metric):
        df = self.data[0]

        for column in [cl for cl in df.columns if metric in cl]:
            fig, ax = plt.subplots(1, 1, dpi=300, figsize=(6,4))
            ax = self.var_lineplot(ax, column, metric)
            ax.legend(self.labels)
            plt.title(column.split("_")[-2])
            pdir = f"{self.output_dir}/plots/{self.logger_metrics[metric]}/{column}"

            os.makedirs(pdir, exist_ok=True)

            plt.savefig(f"{pdir}/output.png", dpi=300)
            plt.close()

    def plot_metrics(self):

        for metric in self.logger_metrics.keys():
            self.lineplot(metric)
            


if __name__ == "__main__":
    exp_dir = "/e/home/jusers/brunstein1/jupiter/repositories/geoarches/geoarches/logs/experiments"
    vlogs = VisualizeLogs(
        exp_dir=exp_dir,
        modelnames=[
            "archesW-s2s-0",
            "archesW-s2s-1",
            "archesW-s2s-2x2-0",
            "archesW-s2s-2x2-1",
            "archesW-s2s-2x2-LE-0",
            "archesW-s2s-2x2-LE-1",

        ],
        apply_smoothing=True,
        ema_weight=0.5

    )

    vlogs.plot_metrics()

        
        