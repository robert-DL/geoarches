"""Compute percentiles per latitude/longitude across a given time range.

Computes both high (99.99th, 99.9th and 99th) and low (0.01st, 0.1st and 1st) climatological percentiles.
Used for brier score computation.
"""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from geoarches.dataloaders import era5

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_data_path",
        type=str,
        default="data/era5_240/full",
        help="Path holding data to compute quantiles on.",
    )
    parser.add_argument(
        "--input_data_file_extension",
        type=str,
        default=".nc",
        choices={".nc", ".zarr"},
        help="File extension for data to compute quantiles on",
    )
    parser.add_argument(
        "--save_path",
        default="src/geoarches/stats/era5-quantiles-2016_2022.nc",
        type=str,
        help="Full filepath to save quantiles netcdf to.",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        default=["2016", "2017", "2018", "2019", "2020", "2021", "2022"],
        help="Year of groundtruth to calculate quantiles on.",
    )
    parser.add_argument(
        "--vars",
        nargs="*",  # Accepts 0 or more arguments as a list.
        default=era5.arches_default_surface_variables
        + era5.arches_default_level_variables
        + ["10m_wind_speed", "wind_speed"],
        help="Variables in dataset to compute quantiles over. Set to empty to compute over all variables.",
    )
    args = parser.parse_args()

    # 6-hourly data within year range.
    filename_filter = lambda x: any(substring in x for substring in args.years)
    files = sorted(
        [
            str(x)
            for x in Path(args.input_data_path).glob(f"*{args.input_data_file_extension}")
            if filename_filter(x.name)
        ]
    )
    print(f"Computing quantiles over {len(files)} files.")

    engine = "netcdf4" if args.input_data_file_extension == ".nc" else "zarr"
    q = np.array([0.01, 0.1, 1, 99, 99.9, 99.99]) / 100

    # Loading every requested variable's full multi-year time series into one
    # chunk at once (as a single open_mfdataset+quantile call would) is
    # memory-heavy enough to get OOM-killed for the larger (level, 13-level)
    # variables over several years. Process one variable at a time instead --
    # peak memory is then bounded by the single largest variable rather than
    # the sum of all of them -- and merge the per-variable quantile results
    # at the end.
    vars_to_process = list(args.vars) if args.vars else None
    if vars_to_process is None:
        with xr.open_dataset(files[0]) as sample:
            vars_to_process = list(sample.data_vars)

    def _open_var(var, level=None):
        def _preprocess(ds, var=var, level=level):
            out = ds.rename(
                {
                    k: v
                    for k, v in {"valid_time": "time", "pressure_level": "level"}.items()
                    if k in ds.dims
                }
            )
            out = out[[var]]
            if level is not None:
                out = out.sel(level=[level])
            return out

        ds = xr.open_mfdataset(
            files, concat_dim="time", combine="nested", preprocess=_preprocess, engine=engine
        )
        # Rechunk time into a single chunk (Required for quantile computation).
        return ds.chunk({"time": -1})

    with xr.open_dataset(files[0]) as sample:
        var_levels = {
            var: list(sample[var]["pressure_level" if "pressure_level" in sample[var].dims else "level"].values)
            if var in sample and any(d in sample[var].dims for d in ("pressure_level", "level"))
            else None
            for var in vars_to_process
        }

    quantile_datasets = []
    for var in vars_to_process:
        levels = var_levels.get(var)
        if levels is None:
            # Surface variable: one (lat, lon, time) field fits comfortably in memory.
            print(f"Computing quantiles for {var} ...")
            quantile_datasets.append(_open_var(var).quantile(q, dim="time").compute())
        else:
            # Level variable: even a single variable's full multi-year time series
            # across all pressure levels at once was enough to get OOM-killed over
            # a long (35-year) year range. Loop one pressure level at a time too --
            # peak memory is then bounded by a single (lat, lon, time) field -- and
            # concat the per-level results back together for this variable.
            per_level = []
            for level in levels:
                print(f"Computing quantiles for {var} @ level {level} ...")
                per_level.append(_open_var(var, level=level).quantile(q, dim="time").compute())
            quantile_datasets.append(xr.concat(per_level, dim="level"))

    quantile_ds = xr.merge(quantile_datasets)
    quantile_ds.to_netcdf(args.save_path)
