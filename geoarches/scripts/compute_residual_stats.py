#!/usr/bin/env python3
"""
Computes forecast-error statistics (prediction - ERA5 truth) for a finished
encode_dataset.py step, using the same mean/std/diff_std schema as the
normalization stats files in geoarches/stats/*.nc (see utils/normalization.py:
NormalizationStatistics, which loads exactly a (statistic, [level]) DataArray
per variable with statistic in {"mean", "std", "diff_std"}).

For each variable (and pressure level, where applicable), pooled over every
year/timestep/lat/lon in [--start-year, --end-year]:
    mu         = mean(pred - era5)                                  (bias)
    sigma      = std(pred - era5)                                   (error spread)
    sigma_diff = std(shift(pred - era5, 1) - (pred - era5))         (day-to-day error autocorrelation)

The shift is applied per-year (per input file) and drops the undefined
Jan-1 boundary value (rather than wrapping Dec-31 into Jan-1 like `.roll`
would), so no cross-year pair is fabricated.

Statistics are pooled online (running sum / sum-of-squares) across years so at
most one prediction file and one ERA5 truth file are held in memory at a time.
NaNs (e.g. land-masked ocean variables) are excluded via skipna. Within a
year, all variables' reductions are computed in a single `dask.compute()`
call so pred/era5 chunks are read from disk once per year instead of once per
reduction.

For variables in --log-variables (default: total_precipitation, olr -- the
variables geoarches.dataloaders.era5.Era5Dataset log-transforms via
apply_log_transform), pred and truth are both first mapped through
log(x + LOG_EPS) -- the same transform and epsilon used there -- before
differencing. This matters because the normalization stats in norm_file (e.g.
stats/daily_norm_stats_w_log.nc) are themselves computed on log-transformed
values for these variables (state/next_state go through apply_log_transform()
before normalize() in Era5Dataset.convert_to_tensordict()); a diff_std left in
physical units for these variables would silently mismatch the data_std it
gets divided into by NormalizationStatistics.compute_state_scaler().

Usage:
    python compute_residual_stats.py <data_dir> <era5_truth_dir> <out.nc> \\
        [--start-year 1979] [--end-year 2013] [--n-threads 8] \\
        [--log-variables total_precipitation olr]

Example:
    python compute_residual_stats.py \\
        data/output/preds/awm-s2s-2x2-doy/step01 \\
        data/era5_1x1_daily/full \\
        data/output/preds/awm-s2s-2x2-doy/stats/awm-s2s-2x2-doy_step01_residual_stats_1979-2013.nc \\
        --start-year 1979 --end-year 2013
"""

import argparse
import re
import time
from pathlib import Path

import dask
import numpy as np
import xarray as xr

DIM_RENAME = {"valid_time": "time", "pressure_level": "level"}
REDUCE_DIMS = ("time", "latitude", "longitude")

# Must match geoarches.dataloaders.era5.Era5Dataset.apply_log_transform's epsilon.
LOG_EPS = 1.0e-5
DEFAULT_LOG_VARIABLES = ["total_precipitation", "olr"]


def _year_from_pred_filename(path: Path) -> int:
    m = re.search(r"_(\d{4})(?:_\d{2}h)?\.nc$", path.name)
    if not m:
        raise ValueError(f"Could not parse year from prediction filename: {path.name}")
    return int(m.group(1))


def _load_era5_year(era5_dir: Path, year: int, variables: list[str], levels) -> xr.Dataset:
    path = era5_dir / f"era5_{year}_valid.nc"
    ds = xr.open_dataset(path, chunks="auto")
    ds = ds.rename({k: v for k, v in DIM_RENAME.items() if k in ds.dims or k in ds.coords})
    ds = ds[variables]
    if "level" in ds.dims:
        ds = ds.sel(level=levels)
    return ds


def _new_accumulator(has_level: bool, n_levels: int) -> dict:
    shape = (n_levels,) if has_level else ()
    return dict(
        sum=np.zeros(shape, dtype=np.float64),
        sumsq=np.zeros(shape, dtype=np.float64),
        dsum=np.zeros(shape, dtype=np.float64),
        dsumsq=np.zeros(shape, dtype=np.float64),
        n=np.zeros(shape, dtype=np.float64),
        dn=np.zeros(shape, dtype=np.float64),
        has_level=has_level,
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("era5_dir", type=Path)
    parser.add_argument("out_file", type=Path)
    parser.add_argument(
        "--start-year", type=int, default=None, help="First calendar year to include (inclusive)."
    )
    parser.add_argument(
        "--end-year", type=int, default=None, help="Last calendar year to include (inclusive)."
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        default=8,
        help="Dask threaded-scheduler worker count used for the per-year reduction.",
    )
    parser.add_argument(
        "--log-variables",
        type=str,
        nargs="*",
        default=DEFAULT_LOG_VARIABLES,
        help=(
            "Variables to log-transform (log(x + %.0e)) before differencing, matching "
            "Era5Dataset.apply_log_transform. Pass an empty list to disable." % LOG_EPS
        ),
    )
    args = parser.parse_args()
    log_variables = set(args.log_variables)

    pred_files = sorted(args.data_dir.glob("*.nc"))
    if not pred_files:
        raise SystemExit(f"No prediction files found in {args.data_dir}")
    pred_files = [
        p
        for p in pred_files
        if (args.start_year is None or _year_from_pred_filename(p) >= args.start_year)
        and (args.end_year is None or _year_from_pred_filename(p) <= args.end_year)
    ]
    if not pred_files:
        raise SystemExit(
            f"No prediction files in year range [{args.start_year}, {args.end_year}] in {args.data_dir}"
        )

    with xr.open_dataset(pred_files[0]) as sample:
        surface_vars = [v for v in sample.data_vars if "level" not in sample[v].dims]
        level_vars = [v for v in sample.data_vars if "level" in sample[v].dims]
        levels = sample["level"].values

    accum = {v: _new_accumulator(False, len(levels)) for v in surface_vars}
    accum.update({v: _new_accumulator(True, len(levels)) for v in level_vars})

    dask.config.set(scheduler="threads", num_workers=args.n_threads)
    all_vars = surface_vars + level_vars

    for pred_path in pred_files:
        year = _year_from_pred_filename(pred_path)
        t0 = time.time()
        print(f"Processing {year} ...", flush=True)
        with xr.open_dataset(pred_path, chunks="auto") as pred_ds:
            with _load_era5_year(args.era5_dir, year, all_vars, levels) as era5_ds:
                era5_ds = era5_ds.sel(time=pred_ds["time"])

                # Build every variable's lazy reductions up front so the final
                # dask.compute() call below reads each pred/era5 chunk from
                # disk exactly once for the whole year, instead of once per
                # reduction (6x fewer file reads than a naive per-quantity
                # .compute()).
                lazy = {}
                for var in all_vars:
                    pred_var, era5_var = pred_ds[var], era5_ds[var]
                    if var in log_variables:
                        pred_var = np.log(pred_var + LOG_EPS)
                        era5_var = np.log(era5_var + LOG_EPS)
                    residual = (pred_var - era5_var).astype(np.float64)
                    # Drop the Jan-1 boundary (undefined prev-day value) instead
                    # of wrapping Dec-31 into it, so no cross-year pair leaks in.
                    diff = residual.shift(time=1) - residual
                    diff = diff.isel(time=slice(1, None))

                    lazy[var] = (
                        residual.notnull().sum(dim=REDUCE_DIMS),
                        residual.sum(dim=REDUCE_DIMS, skipna=True),
                        (residual**2).sum(dim=REDUCE_DIMS, skipna=True),
                        diff.notnull().sum(dim=REDUCE_DIMS),
                        diff.sum(dim=REDUCE_DIMS, skipna=True),
                        (diff**2).sum(dim=REDUCE_DIMS, skipna=True),
                    )

                (results,) = dask.compute(lazy)

                for var in all_vars:
                    n, s, s2, dn, ds_, ds2 = results[var]
                    a = accum[var]
                    a["n"] += n.values.astype(np.float64)
                    a["sum"] += s.values
                    a["sumsq"] += s2.values
                    a["dn"] += dn.values.astype(np.float64)
                    a["dsum"] += ds_.values
                    a["dsumsq"] += ds2.values
        print(f"  {year} done in {time.time() - t0:.1f}s", flush=True)

    data_vars = {}
    for var, a in accum.items():
        mean = a["sum"] / a["n"]
        std = np.sqrt(np.clip(a["sumsq"] / a["n"] - mean**2, 0, None))
        dmean = a["dsum"] / a["dn"]
        diff_std = np.sqrt(np.clip(a["dsumsq"] / a["dn"] - dmean**2, 0, None))

        stat = np.stack([mean, std, diff_std], axis=0).astype(np.float32)
        dims = ("statistic", "level") if a["has_level"] else ("statistic",)
        data_vars[var] = (dims, stat)

    out_ds = xr.Dataset(
        data_vars,
        coords={"statistic": ["mean", "std", "diff_std"], "level": levels},
    )
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    out_ds.to_netcdf(args.out_file)
    print(f"Wrote {args.out_file}")


if __name__ == "__main__":
    main()
