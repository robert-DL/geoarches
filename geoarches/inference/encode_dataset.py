"""
encode_dataset.py
=================
Run inference with one or more models and store encoded / multi-step forecast
outputs as NetCDF files, one file per year × initialisation-hour × forecast step.

Parallelisation
---------------
Run one process per subset of initialisation hours to exploit data parallelism::

    python encode_dataset.py \\
        --uids archesweather-m \\
        --target-path data/era5_240_pred \\
        --encode-hours 0  --forecast-steps 1,4 &
    python encode_dataset.py \\
        --uids archesweather-m \\
        --target-path data/era5_240_pred \\
        --encode-hours 6  --forecast-steps 1,4 &
    python encode_dataset.py \\
        --uids archesweather-m \\
        --target-path data/era5_240_pred \\
        --encode-hours 12 --forecast-steps 1,4 &
    python encode_dataset.py \\
        --uids archesweather-m \\
        --target-path data/era5_240_pred \\
        --encode-hours 18 --forecast-steps 1,4 &

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
import re
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


def collate_fn(lst):
    return {k: torch.stack([x[k] for x in lst]) for k in lst[0]}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description=(
        "Encode an ERA5-like dataset using one or more forecasting models and "
        "optionally store outputs at multiple forecast-step lead times."
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
    help="Process only ~20 batches (quick smoke-test).",
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
    help=(
        "Comma-separated initialisation hours processed by this worker. "
        "Run one process per hour (e.g. '--encode-hours 0') to parallelise."
    ),
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
args = parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Parse arguments & setup
# ─────────────────────────────────────────────────────────────────────────────

torch.set_grad_enabled(False)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

model_uids: list[str] = [u for u in args.uids.split(",") if u]
encode_hours: set[int] = {int(h) for h in args.encode_hours.split(",") if h}
forecast_steps_set: set[int] = {int(s) for s in args.forecast_steps.split(",") if s}
max_steps: int = max(forecast_steps_set)


# ─────────────────────────────────────────────────────────────────────────────
# Load model(s)  – must happen before path setup so we can read cfg
# ─────────────────────────────────────────────────────────────────────────────

if len(model_uids) > 1:
    module = AvgModule(model_uids).to(device).eval()
    cfg = module.cfg
else:
    module, cfg = load_module(model_uids[0])
    module.to(device).eval()


# ─────────────────────────────────────────────────────────────────────────────
# Derive input path & dataset name from model config
# ─────────────────────────────────────────────────────────────────────────────

input_path = Path(cfg.dataloader.dataset.path).resolve()
dataset_name = _extract_dataset_name(input_path)
output_root = Path(args.target_path).resolve()

print(f"Input    : {input_path}  (from model config)")
print(f"Dataset  : {dataset_name}")
print(f"Output   : {output_root}")
print(f"Hours    : {sorted(encode_hours)}")
print(f"Steps    : {sorted(forecast_steps_set)}  (max {max_steps})")


# Create (or reset) one subdirectory per forecast step
step_dirs: dict[int, Path] = {}
for s in sorted(forecast_steps_set):
    d = output_root / f"step{s:02d}"
    if d.exists() and args.force:
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)
    step_dirs[s] = d


# ─────────────────────────────────────────────────────────────────────────────
# Check for already-processed files and determine resume year
# ─────────────────────────────────────────────────────────────────────────────

# Pattern to extract the year from filenames like  <name>_pred_<year>_<HH>h.nc
_year_re = re.compile(r"_(\d{4})_\d{2}h\.nc$")

max_existing_year: int | None = None
for _s, _d in step_dirs.items():
    for _h in encode_hours:
        for _nc in _d.glob(f"*_pred_*_{_h:02d}h.nc"):
            _m = _year_re.search(_nc.name)
            if _m:
                _y = int(_m.group(1))
                if max_existing_year is None or _y > max_existing_year:
                    max_existing_year = _y

if max_existing_year is not None:
    print(
        f"Resume   : existing files found up to year {max_existing_year}. "
        f"Processing will start from year > {max_existing_year}."
    )
else:
    print("Resume   : no existing files found, starting from the beginning.")


# ─────────────────────────────────────────────────────────────────────────────
# Create dataset and dataloader
# ─────────────────────────────────────────────────────────────────────────────

# The dataset path is already set in the config; no override needed.
ds = instantiate(
    cfg.dataloader.dataset,
    cfg.stats,
    domain="all",
)

# infer lead time from era5 dataset
if hasattr(ds, "lead_time_hours"):
    print(f"Inferred lead time from dataset: {ds.lead_time_hours} hours")
    lead_time_hours = ds.lead_time_hours
else:
    print("No lead time info in dataset; defaulting to 24 hours.")
    lead_time_hours = 24

# Restrict dataset range to skip already-processed years
if max_existing_year is not None:
    start_time = np.datetime64(f"{max_existing_year + 1}-01-01T00:00:00")
    if hasattr(ds, "load_prev") and ds.load_prev:
        start_time = start_time - ds.lead_time_hours * np.timedelta64(1, "h")
    ds.set_timestamp_bounds(start_time, None)
    print(f"Resume   : dataset restricted to timestamps >= {start_time}")

dl = torch.utils.data.DataLoader(
    ds,
    batch_size=1,
    num_workers=3,
    shuffle=False,
    collate_fn=collate_fn,
)

# ─────────────────────────────────────────────────────────────────────────────
# Encoding loop
# ─────────────────────────────────────────────────────────────────────────────

# Per-step accumulators: xr_lists[step] holds xr.Dataset items for the
# current initialisation year, to be flushed when the year boundary is crossed.
xr_lists: dict[int, list] = {s: [] for s in forecast_steps_set}
current_year: int | None = None

for i, batch in tqdm(enumerate(dl)):
    # ── Filter by requested initialisation hours ─────────────────────────────
    batch_hour = pd.to_datetime(batch["timestamp"][0], utc=True, unit="s").hour
    if batch_hour not in encode_hours:
        continue

    init_ts_s = int(batch["timestamp"][0].item())
    init_year = pd.to_datetime(init_ts_s, unit="s", utc=True).year

    # ── Year boundary: flush accumulated data to disk ─────────────────────────
    if current_year is not None and init_year > current_year:
        print(f"Saving year {current_year} …")
        _flush_year(xr_lists, current_year, encode_hours, step_dirs, dataset_name)
        xr_lists = {s: [] for s in forecast_steps_set}

    current_year = init_year

    # ── Move batch to device ──────────────────────────────────────────────────
    cur = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}

    # ── Iterative multi-step forecasting ─────────────────────────────────────
    #
    # cur["timestamp"]  = timestamp of the *current input* state (Unix s, shape [1])
    # out               = normalised prediction of the *next* state (+6 h)
    #
    # At each step we:
    #   1. Roll the batch state forward (step > 1 only)
    #   2. Run one forward pass
    #   3. If this step index is in forecast_steps_set, denormalise and store
    out = None
    with torch.no_grad():
        for step in range(1, max_steps + 1):
            if step > 1:
                # Previous output becomes the new input state (still normalised)
                if "prev_state" in cur:
                    cur["prev_state"] = cur["state"]
                cur["state"] = out
                # Advance timestamp by one step (lead_time_hours)
                cur["timestamp"] = cur["timestamp"] + lead_time_hours * 3600
                _update_batch_temporal(cur, int(cur["timestamp"][0].item()), device)
                # NOTE: if the model uses dynamic forcings, load the appropriate
                #       forcing slice for the new timestamp here (cur["forcings"] = ...).
            out = module.forward(
                cur
            )  # normalised prediction at cur["timestamp"] + lead_time_hours

            if step in forecast_steps_set:
                # The predicted state's timestamp is cur["timestamp"] + lead_time_hours
                pred_ts = cur["timestamp"] + lead_time_hours * 3600
                denorm_out = ds.denormalize(out)
                xr_lists[step].append(ds.convert_to_xarray(denorm_out, pred_ts))

        if args.debug and i >= 20:
            break

# ── Final flush ───────────────────────────────────────────────────────────────
if current_year is not None:
    print(f"Saving year {current_year} (final) …")
    _flush_year(xr_lists, current_year, encode_hours, step_dirs, dataset_name)

print("Done.")
