"""
encode_dataset.py
=================
Run inference with one or more models and store encoded / multi-step forecast
outputs as NetCDF files, one file per year × initialisation-hour × forecast step.

Multi-GPU parallelisation
-------------------------
When multiple CUDA devices are available the script automatically splits the
dataset years across GPUs using ``torch.multiprocessing.spawn``.  For example,
if the dataset spans 2000-2019 and 4 GPUs are present each GPU processes a
contiguous block of ~5 years:

    GPU 0: 2000-2004
    GPU 1: 2005-2009
    GPU 2: 2010-2014
    GPU 3: 2015-2019

To encode with a single GPU (or CPU), just run normally::

    python encode_dataset.py --uids archesweather-m \\
        --target-path data/era5_240_pred --forecast-steps 1,4

Resume support
--------------
If a run is interrupted, re-running the same command resumes automatically.
Each GPU independently checks which years in its slice already have complete
output files and skips them.  A year is considered complete when *all*
expected ``step × hour`` NetCDF files exist for it.

The input dataset path is read from the model config
(``cfg.dataloader.dataset.path``).  The dataset name is derived from
that path's files (``<name>_*`` naming convention).

Output layout
-------------
``<target_path>/
    step01/<dataset_name>_pred_<year>_00h.nc
    step01/<dataset_name>_pred_<year>_06h.nc
    ...
    step04/<dataset_name>_pred_<year>_00h.nc
    ...``
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from hydra.utils import instantiate
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from geoarches.lightning_modules.base_module import AvgModule, load_module

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _extract_dataset_name(input_path: Path) -> str:
    """Infer the dataset name from the common prefix of *.nc files in *input_path*.

    Input files are expected to follow the ``<name>_*`` naming convention.
    Example: ``era5_240_2020.nc``, ``era5_240_2021.nc`` → ``era5_240``.
    Falls back to the directory name when no files can be found.
    """
    files = sorted(input_path.glob("*.nc"))
    if not files:
        files = sorted(input_path.glob("**/*.nc"))
    if not files:
        return input_path.name  # last-resort fallback

    # os.path.commonprefix operates character-by-character; strip trailing "_"
    prefix = os.path.commonprefix([f.stem for f in files]).rstrip("_")
    return prefix or input_path.name


def _update_batch_temporal(batch: dict, ts_seconds: int, device) -> None:
    """Update the temporal-metadata fields of *batch* in-place for *ts_seconds*."""
    ts = pd.Timestamp(ts_seconds, unit="s", tz="UTC")
    batch["hour_of_day"] = torch.tensor([ts.hour], dtype=torch.int64, device=device)
    batch["day_of_month"] = torch.tensor([ts.day], dtype=torch.int64, device=device)
    batch["day_of_year"] = torch.tensor([ts.dayofyear], dtype=torch.int64, device=device)
    batch["month"] = torch.tensor([ts.month], dtype=torch.int64, device=device)


def _flush_year(
    xr_lists: dict,
    year: int,
    encode_hours: set,
    step_dirs: dict,
    dataset_name: str,
) -> None:
    """Concatenate accumulated xr.Datasets and write one NetCDF per step × hour."""
    for s, xr_items in xr_lists.items():
        if not xr_items:
            continue
        xr_ds = xr.concat(xr_items, dim="time")
        for h in sorted(encode_hours):
            hour_ds = xr_ds.sel(time=(xr_ds.time.dt.hour == h))
            if hour_ds.time.size == 0:
                continue
            fname = f"{dataset_name}_pred_{year}_{h:02d}h.nc"
            out_file = step_dirs[s] / fname
            if not out_file.exists():
                hour_ds.to_netcdf(
                    out_file,
                    encoding={"time": {"units": "hours since 2000-01-01"}},
                )


def _fill_land_points(prediction, nan_mask, nan_replacement, variables):
    """Fill land points in a normalised TensorDict prediction.

    Called after every forward pass so that land grid points carry physically
    consistent values when the output is recycled as the next input state.

    Parameters
    ----------
    prediction : TensorDict
        Normalised model output (no NaN values).
    nan_mask : TensorDict or None
        Same structure as *prediction*; ``True`` marks land (NaN) positions.
    nan_replacement : str
        ``"zero"``          – set all land points to 0 (normalised).
        ``"lat_mean_sic_zero"`` – fill land points with the zonal mean of
        valid (ocean) values at each latitude row, then override SST and
        SIC land points with 0.
    variables : dict
        ``ds.variables`` dict for variable-index look-ups.
    """
    if nan_mask is None:
        return prediction

    if nan_replacement == "zero":
        return prediction.apply(lambda p, m: p.masked_fill(m.bool(), 0.0), nan_mask)

    if nan_replacement == "lat_mean_sic_zero":
        idx_sic = variables["surface"].index("sea_ice_cover")
        idx_sst = variables["surface"].index("sea_surface_temperature")

        def _fill_lat_mean(pred_tensor, mask_tensor):
            # Mask land temporarily so they are excluded from the lon-axis mean.
            pred_with_nan = pred_tensor.masked_fill(mask_tensor.bool(), float("nan"))
            lat_mean = torch.nanmean(pred_with_nan, dim=-1, keepdim=True)
            return torch.where(mask_tensor.bool(), lat_mean.expand_as(pred_tensor), pred_tensor)

        prediction = prediction.apply(_fill_lat_mean, nan_mask)

        # SST and SIC should be zero over land.
        surface = prediction["surface"]  # (batch, n_vars, 1, lat, lon)
        mask_surface = nan_mask["surface"]  # (batch, n_vars, 1, lat, lon)
        surface[:, idx_sic] = surface[:, idx_sic].masked_fill(mask_surface[:, idx_sic].bool(), 0.0)
        surface[:, idx_sst] = surface[:, idx_sst].masked_fill(mask_surface[:, idx_sst].bool(), 0.0)
        prediction["surface"] = surface
        return prediction

    raise ValueError(f"Unsupported nan_replacement strategy: {nan_replacement!r}")


def _apply_nan_mask(prediction, nan_mask):
    """Re-insert NaN at land positions in a denormalised TensorDict.

    Should be called on CPU tensors just before ``convert_to_xarray``.
    """
    if nan_mask is None:
        return prediction
    return prediction.apply(
        lambda p, m: p.masked_fill(m.bool().cpu(), float("nan")),
        nan_mask,
    )


def collate_fn(lst):
    return {k: torch.stack([x[k] for x in lst]) for k in lst[0]}


def _get_all_years(ds) -> list[int]:
    """Return a sorted list of unique calendar years present in *ds* timestamps."""
    return sorted({int(pd.Timestamp(t[-1]).year) for t in ds.timestamps})


def _split_years(years: list[int], rank: int, world_size: int) -> list[int]:
    """Assign a contiguous slice of *years* to *rank*.

    Years are distributed as evenly as possible; the first ``len(years) %
    world_size`` ranks each receive one extra year.
    """
    n = len(years)
    base, remainder = divmod(n, world_size)
    start = rank * base + min(rank, remainder)
    end = start + base + (1 if rank < remainder else 0)
    return years[start:end]


def _year_is_done(
    year: int,
    step_dirs: dict,
    encode_hours: set,
    dataset_name: str,
) -> bool:
    """Return ``True`` iff all expected output files for *year* already exist.

    A year is considered complete when every combination of forecast step and
    initialisation hour has a corresponding NetCDF file on disk.
    """
    for s, d in step_dirs.items():
        for h in encode_hours:
            if not (d / f"{dataset_name}_pred_{year}_{h:02d}h.nc").exists():
                return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Encode an ERA5-like dataset using one or more forecasting models and "
            "optionally store outputs at multiple forecast-step lead times. "
            "When multiple CUDA devices are available, the dataset years are "
            "automatically split across GPUs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--target-path",
        required=True,
        help="Directory where all step output subdirectories are written.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove and recreate step output subdirectories before running.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Process only ~20 batches per GPU (quick smoke-test).",
    )
    parser.add_argument(
        "--uids",
        default="",
        type=str,
        help=(
            "Comma-separated model UIDs (checkpoint folder names under 'modelstore/'). "
            "When more than one UID is given, their predictions are averaged."
        ),
    )
    parser.add_argument(
        "--encode-hours",
        default="0,6,12,18",
        type=str,
        help="Comma-separated initialisation hours to encode.",
    )
    parser.add_argument(
        "--forecast-steps",
        default="1",
        type=str,
        help=(
            "Comma-separated 1-based forecast-step indices at which to save output datasets. "
            "Each step is one model forward pass (typically 24 h lead time). "
            "Example: --forecast-steps 1,4,8  saves outputs at +24 h, +96 h, and +192 h."
        ),
    )
    parser.add_argument(
        "--nan-replacement",
        default="zero",
        choices=["zero", "lat_mean_sic_zero"],
        help=(
            "Strategy used to fill land grid points in predicted states before "
            "recycling them as the next model input. "
            "'zero' sets all land points to 0 (normalised). "
            "'lat_mean_sic_zero' fills with the zonal mean and zeros SST/SIC on land."
        ),
    )
    return parser


# ─────────────────────────────────────────────────────────────────────────────
# Per-GPU worker
# ─────────────────────────────────────────────────────────────────────────────


def _encode_worker(rank: int, world_size: int, args) -> None:
    """Encode the calendar years assigned to *rank* on ``cuda:{rank}``.

    Each worker independently:
    1. Determines which years in its assigned slice are already fully encoded.
    2. Restricts the dataset to the remaining years (plus the lead-time
       overlap needed for ``load_prev`` / ``multistep``).
    3. Runs the iterative forecast loop and writes per-year NetCDF files.
    """
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    torch.set_grad_enabled(False)

    # ── Parsed args ───────────────────────────────────────────────────────────
    model_uids: list[str] = [u for u in args.uids.split(",") if u]
    encode_hours: set[int] = {int(h) for h in args.encode_hours.split(",") if h}
    forecast_steps_set: set[int] = {int(s) for s in args.forecast_steps.split(",") if s}
    max_steps: int = max(forecast_steps_set)

    # ── Load model ────────────────────────────────────────────────────────────
    if len(model_uids) > 1:
        module = AvgModule(model_uids).to(device).eval()
        cfg = module.cfg
    else:
        module, cfg = load_module(model_uids[0])
        module.to(device).eval()

    # ── Paths & step dirs ─────────────────────────────────────────────────────
    input_path = Path(cfg.dataloader.dataset.path).resolve()
    dataset_name = _extract_dataset_name(input_path)
    output_root = Path(args.target_path).resolve()

    step_dirs: dict[int, Path] = {}
    for s in sorted(forecast_steps_set):
        d = output_root / f"step{s:02d}"
        d.mkdir(parents=True, exist_ok=True)
        step_dirs[s] = d

    if rank == 0:
        print(f"Input    : {input_path}  (from model config)")
        print(f"Dataset  : {dataset_name}")
        print(f"Output   : {output_root}")
        print(f"Hours    : {sorted(encode_hours)}")
        print(f"Steps    : {sorted(forecast_steps_set)}  (max {max_steps})")
        print(f"Devices  : {world_size}")

    # ── Instantiate full dataset (unfiltered) to discover all years ───────────
    ds = instantiate(cfg.dataloader.dataset, cfg.stats, domain="all")
    lead_time_hours: int = getattr(ds, "lead_time_hours", 24)
    multistep: int = getattr(ds, "multistep", 0)
    load_prev: bool = bool(getattr(ds, "load_prev", False))

    if rank == 0:
        print(
            f"Lead time: {lead_time_hours} h  |  multistep: {multistep}  |  load_prev: {load_prev}"
        )

    # ── Year splitting ────────────────────────────────────────────────────────
    all_years = _get_all_years(ds)
    if not all_years:
        print(f"[GPU {rank}] Dataset has no timestamps. Exiting.")
        return

    # Filter to not-yet-done years *before* splitting across workers so that
    # each GPU receives an equal share of the remaining work, not of the full
    # historical range (which would leave most GPUs idle when resuming).
    remaining_years = [
        y for y in all_years if not _year_is_done(y, step_dirs, encode_hours, dataset_name)
    ]
    if not remaining_years:
        print(f"[GPU {rank}] All years already encoded. Nothing to do.")
        return

    if rank == 0:
        done_count = len(all_years) - len(remaining_years)
        if done_count:
            print(
                f"Resume   : {done_count} year(s) already done, "
                f"{len(remaining_years)} remaining "
                f"({remaining_years[0]}–{remaining_years[-1]})."
            )

    worker_years = _split_years(remaining_years, rank, world_size)
    if not worker_years:
        print(f"[GPU {rank}] No years assigned. Exiting.")
        return

    print(
        f"[GPU {rank}] Assigned {len(worker_years)} year(s): {worker_years[0]}–{worker_years[-1]}"
    )

    todo_years_set: set[int] = set(worker_years)

    # ── Restrict dataset to this worker's time window ─────────────────────────
    # low  = start of first assigned year.  Pulled back by lead_time_hours when
    #        load_prev=True so the dataset can serve the prev_state for the
    #        very first batch of that year.
    # high = start of the year *after* the last assigned year.  Extended by
    #        multistep * lead_time_hours so next_state is available for the
    #        final timesteps (required by Era5Forecast.__len__ / __getitem__).
    low = np.datetime64(f"{worker_years[0]}-01-01T00:00:00")
    high = np.datetime64(f"{worker_years[-1] + 1}-01-01T00:00:00")

    if load_prev:
        low = low - lead_time_hours * np.timedelta64(1, "h")
    if multistep > 0:
        high = high + multistep * lead_time_hours * np.timedelta64(1, "h")

    ds.set_timestamp_bounds(low, high)
    print(f"[GPU {rank}] Timestamp bounds: [{low}, {high})")

    # ── Build a Subset covering exactly the todo_years ────────────────────────
    # Era5Forecast.__getitem__(i) reads the sample timestamp from
    # id2pt[i + load_prev_offset], where load_prev_offset accounts for the
    # prev_state entry prepended to the timestamp list.  We collect the
    # logical indices (0 … len(ds)-1) whose actual sample year is in
    # todo_years_set.  Using a Subset guarantees the DataLoader iterates
    # precisely those samples — no reliance on set_timestamp_bounds being
    # reflected correctly in __len__, and no continue-based filtering.
    load_prev_offset = int(load_prev) * lead_time_hours // ds.timedelta
    todo_indices = [
        i
        for i in range(len(ds))
        if pd.Timestamp(ds.id2pt[i + load_prev_offset][2]).year in todo_years_set
    ]
    print(f"[GPU {rank}] {len(todo_indices)} sample(s) to encode.")

    # ── DataLoader ────────────────────────────────────────────────────────────
    # multiprocessing_context='fork': the dataset stores a non-picklable
    # lambda (filename_filter).  Fork-based workers inherit the parent
    # process memory without pickling, so the lambda is accessible.
    # This is safe on Linux (the standard GPU cluster environment).
    dl = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, todo_indices),
        batch_size=1,
        num_workers=4,
        multiprocessing_context="fork",
        shuffle=False,
        collate_fn=collate_fn,
    )

    # ── Encoding loop ─────────────────────────────────────────────────────────
    xr_lists: dict[int, list] = {s: [] for s in forecast_steps_set}
    current_year: int | None = None

    for i, batch in tqdm(enumerate(dl), desc=f"GPU {rank}", total=len(dl)):
        # Filter by initialisation hour
        batch_hour = pd.to_datetime(batch["timestamp"][0], utc=True, unit="s").hour
        print("Datetime:", pd.to_datetime(batch["timestamp"][0], unit="s").tz_localize(None))
        if batch_hour not in encode_hours:
            continue

        init_ts_s = int(batch["timestamp"][0].item())
        init_year = pd.to_datetime(init_ts_s, unit="s", utc=True).year

        # Year boundary: flush accumulated data to disk
        if current_year is not None and init_year > current_year:
            print(f"[GPU {rank}] Saving year {current_year} …")
            _flush_year(xr_lists, current_year, encode_hours, step_dirs, dataset_name)
            xr_lists = {s: [] for s in forecast_steps_set}

        current_year = init_year

        # Move batch to device
        cur = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
        nan_mask = cur.get("nan_mask", None)

        # Iterative multi-step forecast
        out = None
        with torch.no_grad():
            for step in range(1, max_steps + 1):
                if step > 1:
                    if "prev_state" in cur:
                        cur["prev_state"] = cur["state"]
                    cur["state"] = out
                    cur["timestamp"] = cur["timestamp"] + lead_time_hours * 3600
                    _update_batch_temporal(cur, int(cur["timestamp"][0].item()), device)
                    # NOTE: if the model uses dynamic forcings, load the
                    # appropriate forcing slice for the new timestamp here.

                out = module.forward(cur)

                # Fill land grid points so recycling as the next input is
                # physically consistent.
                out = _fill_land_points(out, nan_mask, args.nan_replacement, ds.variables)

                if step in forecast_steps_set:
                    pred_ts = cur["timestamp"] + lead_time_hours * 3600
                    denorm_out = ds.denormalize(out.cpu())
                    denorm_out = _apply_nan_mask(denorm_out, nan_mask)
                    xr_lists[step].append(ds.convert_to_xarray(denorm_out, pred_ts))

        if args.debug and i >= 20:
            break

    # Final flush
    if current_year is not None:
        print(f"[GPU {rank}] Saving year {current_year} (final) …")
        _flush_year(xr_lists, current_year, encode_hours, step_dirs, dataset_name)

    print(f"[GPU {rank}] Done.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    args = _build_arg_parser().parse_args()

    # Handle --force before spawning to avoid races between workers.
    if args.force:
        output_root = Path(args.target_path).resolve()
        forecast_steps_set = {int(s) for s in args.forecast_steps.split(",") if s}
        for s in sorted(forecast_steps_set):
            d = output_root / f"step{s:02d}"
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
        print(f"Force    : cleared step subdirectories under {output_root}")

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"Detected {num_gpus} CUDA devices — spawning {num_gpus} workers.")
        torch.multiprocessing.spawn(
            _encode_worker,
            args=(num_gpus, args),
            nprocs=num_gpus,
            join=True,
        )
    else:
        # Single GPU or CPU — run directly in the current process.
        _encode_worker(0, 1, args)


if __name__ == "__main__":
    main()
