import itertools
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import xarray as xr
from torch import Tensor
from torchmetrics import Metric


class LabelWrapper(Metric):
    """Wrapper class for extracting metric values into a labelled dictionary.
    Helpful for WandB which needs to log 1D tensors.

    Expects the wrapped metric to return a dictionary holding computed metrics:
        - keys: metric names
        - values: torch tensors with shape (..., *(variable_index))
                  variable_index is passed in with param `variable_indices`

    Returns dictionary of computed metrics:
        - keys: metric name + variable name
        - value: torch tensors with shape (...)

    Warning: this class is not compatible with forward(), only use update() and compute().
    See https://github.com/Lightning-AI/torchmetrics/issues/987#issuecomment-2419846736.

    Example:
        metric = LabelWrapper(EnsembleMetrics(preprocess=preprocess_fn),
                              variable_indices=dict(T2m=(2, 0), U10=(0, 0)),
                              lead_time_hours=24,
                              multistep=2)
        targets, preds = torch.tensor(batch, ..., var, lev, lat, lon), torch.tensor(batch, ..., var, lev, lat, lon)
        metric.update(targets, preds)
        labeled_dict = metric.compute()  # EnsembleMetrics returns {"mse": torch.tensor(... , var, lev) }
        labelled_dict = {"mse_T2m": ..., "mse_T2m_48h": ..., "mse_U10_24h": ..., "mse_U10_48h": ...}

    Args:
        metric: base metric that should be wrapped. It is assumed that the metric outputs a
            dict mapping metric name to tensors that have shape (..., *(variable_index)).
        variable_indices: Mapping from variable name to index (ie. var, lev) into tensor holding computed metric.
                ie. dict(T2m=(2, 0), U10=(0, 0), V10=(1, 0), SP=(3, 0)).
        lead_time_hours: set to explicitly handle metrics computed on predictions from multistep rollout.
                Metrics are computed per lead time (aka. prediction_timedelta).
                FYI when set to None, wrapper with 'dict' output format still handles natively any extra dimensions in targets/preds.
                However, this option labels each timestep separately in output metric dict.
                If lead_time_hours set, assumes that metric tensor shapes are (..., prediction_timedelta, *(variable_index)).
        multistep: set to explicitly handle metrics computed on predictions from multistep rollout.
                Size of prediction_timedelta dimension. See param `lead_time_hours`.
        return_raw_dict: Whether to also return the raw output from the metrics (along with the labelled dict).
    """

    def __init__(
        self,
        metric: Metric,
        variable_indices: Dict[str, tuple],
        lead_time_hours: None | int = None,
        rollout_iterations: None | int = None,
        return_raw_dict: bool = False,
    ):
        super().__init__()
        if not isinstance(metric, Metric):
            raise ValueError(
                f"Expected argument `metric` to be an instance of `torchmetrics.Metric` but got {metric}"
            )
        self.metric = metric
        self.variable_indices = variable_indices

        if bool(lead_time_hours) ^ bool(rollout_iterations):
            raise ValueError(
                f"Need to specify both `lead_time_hours` and `rollout_iterations` or neither. "
                f"Got lead_time_hours: {lead_time_hours} and rollout_iterations: {rollout_iterations}."
            )

        self.lead_time_hours = lead_time_hours
        self.rollout_iterations = rollout_iterations
        self.return_raw_dict = return_raw_dict

    def _convert(self, raw_metric_dict: Dict[str, Tensor]):
        indices = {}
        if self.lead_time_hours:
            # Handle multistep predictions explicitly by separate labels in metric output dict.
            for var, index in self.variable_indices.items():
                for i in range(self.rollout_iterations):
                    lead_time = self.lead_time_hours * (i + 1)
                    indices[f"{var}_{lead_time}h"] = (i, *index)
        else:
            indices = self.variable_indices

        # Label metrics.
        labeled_dict = dict()
        for var, index in indices.items():
            for metric_name, metric in raw_metric_dict.items():
                labeled_dict[f"{metric_name}_{var}"] = metric.__getitem__((..., *index))
        return labeled_dict

    def update(self, *args: Any, **kwargs: Any) -> None:
        self.metric.update(*args, **kwargs)

    def compute(self) -> Dict[str, Tensor]:
        raw_metric_dict = self.metric.compute()
        if self.return_raw_dict:
            return raw_metric_dict, self._convert(raw_metric_dict)
        else:
            return self._convert(raw_metric_dict)

    def reset(self) -> None:
        """Reset metric."""
        self.metric.reset()
        super().reset()


def convert_metric_dict_to_xarray(
    labeled_dict: Dict[str, Tensor], extra_dimensions: List[str] = []
):
    """
    Convert labeled dict with metrics to labeled xarray.

    Example:
        labelled_dict = {"mse_T2m_24h": 1.0, "mse_T2m_48h": 2.0}
        ds = convert_metric_dict_to_xarray(labelled_dict, extra_dimensions=['prediction_timedelta'])

    Args:
        labeled_dict: Mapping to metric tenors. Keys are formated as "<metric>_<var>_<dim1>_<dim2>_..._"
            where the separator between dimensions is an underscore.
        extra_dimensions: list of dimension names, if any extra.
    """

    def _convert_coord(name, value):
        if "timedelta" in name:
            return pd.to_timedelta(value)
        if "bin" in name:
            return int(value)
        else:
            return value

    # Collect coordinates.
    variables = set()
    metrics = set()
    coords = defaultdict(set)
    for label in labeled_dict:
        labels = label.split("_")
        if len(labels) - 2 != len(extra_dimensions):
            raise ValueError(
                f"Expected length of extra_dimensions for key {label} to be: {len(labels) - 2}."
            )
        metrics.add(labels[0])
        variables.add(labels[1])
        for i, dim in enumerate(extra_dimensions):
            coords[dim].add(labels[i + 2])

    dimension_shape = [len(coord) for coord in (variables, *coords.values())]
    # Sort coordinates.
    variables = sorted(list(variables))
    for k, coord in coords.items():
        coords[k] = sorted(list(coord), key=lambda x: _convert_coord(k, x))

    # Aggregate data arrays by variable.
    dimensions = ["variable"] + extra_dimensions
    data_arrays = {}
    for metric in metrics:
        data = []
        for dims in itertools.product(variables, *coords.values()):
            var, other_dims = dims[0], dims[1:]
            other_dims = "_" + "_".join(other_dims) if other_dims else ""
            key = f"{metric}_{var}{other_dims}"
            data.append(labeled_dict[key])
        data = np.array(data).reshape(dimension_shape)
        data_arrays[metric] = (dimensions, data)

    # Prepare coordinates.
    for k, coord in coords.items():
        coords[k] = [_convert_coord(k, x) for x in coord]
    coords["variable"] = variables

    return xr.Dataset(data_vars=data_arrays, coords=coords)
