import torch
from tensordict import TensorDict


class TemporalMean:
    """Class for keeping track of average values of variables.

    Each TemporalMean keeps track of one variable / key in a TensorDict specified at initialization.

    Note that all information on the variable history is stored in the state itself. Thus, an instance of this class
    does not need to be applied in a "chronological order" for keeping track of the mean values.

    Args:
        variable (str): The name of the variable to average.
        period (float): The period over which to compute the means.
        resolution (float): Length of time bins over which mean values are stored in order to reduce memory costs.
        use_incomplete_result (bool, optional): Whether to take means over smaller periods
            for shorter histories. Defaults to True.
    """

    def __init__(
        self,
        variable: str,
        period: float,
        resolution: float,
        use_incomplete_result: bool = True,
        *args,
        **kwargs,
    ):
        self.variable = variable
        self.period = float(period)
        self.n = round(period / resolution)
        self.use_incomplete = use_incomplete_result

        self.dt_bin = self.period / self.n

    def forward(self, concentration: TensorDict, dt: float, dynamic_inputs: TensorDict, *args, **kwargs) -> TensorDict:
        """Calculate the next mean value.

        Args:
            concentration (TensorDict): The state for which the time average should be updated.
            dt (float): Time step of the simulation. Required for time averaging.
            dynamic_inputs (TensorDict): Additional inputs. Has to contain the key "t" for time since simulation start.

        Returns:
            TensorDict: The state with the updated time average value.
        """
        concentration = concentration.clone()
        time = dynamic_inputs["time"]
        value = concentration[self.variable]
        time = time[(...,) + (None,) * (value.ndim - time.ndim)]
        dtype_map = {torch.float32: torch.int32, torch.float64: torch.int64}
        if not self.variable + "_previous_time" in concentration:
            concentration[self.variable + "_previous_time"] = time
            concentration[self.variable + "_bin_end_time"] = time + self.dt_bin
            concentration[self.variable + "_icurrent"] = torch.zeros(
                size=concentration.batch_size + (1,) * (value.ndim - concentration.ndim + 1),
                dtype=dtype_map[value.dtype],
            )
            concentration[self.variable + "_ioldest"] = torch.ones(
                size=concentration.batch_size + (1,) * (value.ndim - concentration.ndim + 1),
                dtype=dtype_map[value.dtype],
            )

            concentration[self.variable + "_previous_value"] = torch.zeros_like(value)
            concentration[self.variable + "_history"] = torch.zeros(
                value.shape + (self.n + 1,), dtype=value.dtype, device=value.device
            )
            concentration[self.variable + "_last_exact_mean"] = torch.zeros_like(value)
            concentration[self.variable + "_mean"] = torch.zeros_like(value)

            concentration[self.variable + "_complete"] = torch.full(
                concentration.batch_size + (1,) * (value.ndim - concentration.ndim), False
            )

            return concentration

        _previous_value = concentration[self.variable + "_previous_value"]
        _previous_time = concentration[self.variable + "_previous_time"]
        _bin_end_time = concentration[self.variable + "_bin_end_time"]
        _icurrent = concentration[self.variable + "_icurrent"]
        _ioldest = concentration[self.variable + "_ioldest"]

        _previous_value = concentration[self.variable + "_previous_value"]
        _history = concentration[self.variable + "_history"]
        _last_exact_mean = concentration[self.variable + "_last_exact_mean"]
        _mean = concentration[self.variable + "_mean"]

        _complete = concentration[self.variable + "_complete"]

        while torch.any(time >= _bin_end_time):  # TODO won't work: time is tensor with batch dim
            mask = time >= _bin_end_time
            dt = _bin_end_time - _previous_time
            w = dt / (time - _previous_time)  # interpolation weight

            # linear-interpolated integral over the last dt interval (trapezoid)
            scale = dt / self.period

            _history = torch.where(
                mask.unsqueeze(-1),
                _history.scatter_add(
                    -1,
                    _icurrent.expand(_history.shape[:-1] + (-1,)),
                    (((1.0 - 0.5 * w) * _previous_value + 0.5 * w * value) * scale).unsqueeze(-1),
                ),
                _history,
            )

            _last_exact_mean = torch.where(
                mask,
                torch.where(
                    _complete,
                    _last_exact_mean
                    - _history.gather(-1, _ioldest.expand(_history.shape[:-1] + (-1,))).squeeze(-1)
                    + _history.gather(-1, _icurrent.expand(_history.shape[:-1] + (-1,))).squeeze(
                        -1
                    ),  # remove oldest bin, add current full bin
                    _last_exact_mean + _history.gather(-1, _icurrent.expand(_history.shape[:-1] + (-1,))).squeeze(-1),
                ),  # only add the newly filled bin
                _last_exact_mean,
            )

            _complete = torch.where(
                mask, torch.where(_complete, _complete, torch.eq(_icurrent.squeeze(-1), self.n - 1)), _complete
            )

            # update previous_value to interpolated value at bin boundary
            _previous_value = torch.where(mask, (1.0 - w) * _previous_value + w * value, _previous_value)
            _previous_time = torch.where(mask, _bin_end_time, _previous_time)

            # advance circular buffer
            _icurrent = torch.where(mask.unsqueeze(-1), _ioldest, _icurrent)
            _ioldest = torch.where(mask.unsqueeze(-1), torch.where(_ioldest > self.n - 1, 0, _ioldest + 1), _ioldest)

            _history = torch.where(
                mask.unsqueeze(-1), _history.scatter(-1, _icurrent.expand(_history.shape[:-1] + (-1,)), 0.0), _history
            )

            _bin_end_time = torch.where(mask, _bin_end_time + self.dt_bin, _bin_end_time)

        # --- 2: Add partial last interval (previous_time → time) ---
        dt = time - _previous_time
        scale = torch.where(dt > 0, 0.5 * dt / self.period, 0)
        _history = _history.scatter(
            -1,
            _icurrent.expand(_history.shape[:-1] + (-1,)),
            _history.gather(-1, _icurrent.expand(_history.shape[:-1] + (-1,)))
            + torch.where(dt > 0, scale * (_previous_value + value), 0).unsqueeze(-1),
        )

        # --- 3: Update mean ---
        if not self.use_incomplete:
            scale = (time - _bin_end_time + self.dt_bin) / self.dt_bin
            _mean = torch.where(
                _complete,
                _last_exact_mean
                + _history.gather(-1, _icurrent.expand(_history.shape[:-1] + (-1,))).squeeze(-1)
                - scale * _history.gather(-1, _ioldest.expand(_history.shape[:-1] + (-1,))).squeeze(-1),
                _mean,
            )
        else:
            mask = torch.logical_and(torch.eq(_previous_time, time), torch.eq(_icurrent.squeeze(-1), 1))
            elapsed = (_icurrent.squeeze(-1) + 1) * self.dt_bin + time - _bin_end_time
            scale = torch.where(_complete, (time - _bin_end_time + self.dt_bin) / self.dt_bin, self.period / elapsed)
            _mean = torch.where(
                _complete,
                _last_exact_mean
                + _history.gather(-1, _icurrent.expand(_history.shape[:-1] + (-1,))).squeeze(-1)
                - scale * _history.gather(-1, _ioldest.expand(_history.shape[:-1] + (-1,))).squeeze(-1),
                torch.where(
                    mask,
                    value,
                    scale
                    * (
                        _last_exact_mean
                        + _history.gather(-1, _icurrent.expand(_history.shape[:-1] + (-1,))).squeeze(-1)
                    ),
                ),
            )

        _previous_time = time
        _previous_value = value

        concentration[self.variable + "_previous_value"] = _previous_value
        concentration[self.variable + "_previous_time"] = _previous_time
        concentration[self.variable + "_bin_end_time"] = _bin_end_time
        concentration[self.variable + "_icurrent"] = _icurrent
        concentration[self.variable + "_ioldest"] = _ioldest

        concentration[self.variable + "_previous_value"] = _previous_value
        concentration[self.variable + "_history"] = _history
        concentration[self.variable + "_last_exact_mean"] = _last_exact_mean
        concentration[self.variable + "_mean"] = _mean

        concentration[self.variable + "_complete"] = _complete

        return concentration
