import torch 
import pandas as pd

from geoarches.dataloaders import era5_constants, nan_util

def apply_nan_mask(x, dataset, fill_nans=False, nan_mask=None):
    if fill_nans:
        x = nan_util.post_norm_interpolate_nans(
            x, "reset_nans", nan_mask=nan_mask
        )
    else:
        x = nan_util.post_norm_interpolate_nans(
                x, dataset.interpolate_input, nan_mask=nan_mask
        )
    return x

def truncate_non_negative(tdict, dataset):
        
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
        variables = dataset.variables[key]
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

def advance_batch(batch, output, dataset, nan_mask):
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
    cout = dataset.invert_log_transform(dataset.denormalize(cout))
    cout = truncate_non_negative(cout, dataset)
    cout = dataset.apply_log_transform(cout)
    cout = dataset.normalize(cout)

    batch["state"] = apply_nan_mask(cout, dataset, nan_mask=nan_mask)

    # Forcings if applicable. load_forcings() internally looks up forcings for
    # (timestamp + lead_time_hours), so we must query it with the timestamp of
    # the state that was just produced (new_datetime) to get forcings for the
    # following step, not the ones already used as input this step.

    if "forcings" in batch.keys():
        new_forcings = dataset.load_forcings(new_datetime.to_numpy())
        new_forcings = dataset.normalize({"forcings": new_forcings})["forcings"]
        new_forcings = apply_nan_mask(new_forcings, dataset)
        new_forcings = new_forcings.unsqueeze(0).to(device)
        batch["forcings"] = new_forcings

    return batch
