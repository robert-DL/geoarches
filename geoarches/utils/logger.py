"""
Custom GeoarchesLogger that circumvents problems with wandb and online logging
when online logging is not available
"""

import os
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

# Metric to 

class VisualizeLogs:
    def __init__(self, exp_dir, modelnames, metrics=["rmse"], output_dir=None):
        super().__init__()

        paths = [exp_dir + f"/{name}" for name in modelnames]
        if output_dir is None:
            self.output_dir = paths[0]
        else:
            self.output_dir = output_dir

        self.data  = [pd.read_csv(f"{p}/metrics.csv") for p in paths]
        
        self.hyparams = []
        self.labels = []
        for p in paths:
            with open(os.path.join(p,"hparams.yaml")) as stream:
                cfg = yaml.safe_load(stream)
                self.hyparams.append(cfg)
                self.labels.append(cfg["name"])
            
        self.metrics = metrics 

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

            df.plot(
                x=x,
                y=clmn, 
                ax=ax, 
                grid=True, 
                logx=False, 
                logy=True, 
                xlabel=x, 
                ylabel=f"{metric}"
            )

        return ax

    def lineplot(self, metric):
        df = self.data[0]
        for column in [cl for cl in df.columns if metric in cl]:
            fig, ax = plt.subplots(1, 1, dpi=300, figsize=(6,4))
            ax = self.var_lineplot(ax, column, metric)
            ax.legend(self.labels)
            os.makedirs(f"{self.output_dir}/plots/{metric}/{column}", exist_ok=True)
            plt.savefig(f"{self.output_dir}/plots/{metric}/{column}/output.png", dpi=300)
            plt.close()

    def plot_metrics(self):

        for metric in self.metrics:
            self.lineplot(metric)
            


if __name__ == "__main__":
    exp_dir = "/e/home/jusers/brunstein1/jupiter/repositories/geoarches/geoarches/logs/experiments"
    vlogs = VisualizeLogs(
        exp_dir=exp_dir,
        modelnames=[
            "archesW-s2s-0",
            "archesW-s2s-1",
            "archesW-s2s-2",
            "archesW-s2s-3",
            "archesW-s2s-2x2-0",
            "archesW-s2s-2x2-1",
            "archesW-s2s-6",
            "archesW-s2s-7"

        ]

    )

    vlogs.plot_metrics()
    

        
        