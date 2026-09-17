from typing import Callable, Optional, Union

import torch
from tensordict import TensorDict


def ode_solver(
    ode_method: str,
    concentration: Union[TensorDict, torch.Tensor],
    dt: Union[float, torch.Tensor],
    dynamic_inputs: TensorDict,
    rhs_function: Callable,
    ppdd_function: Optional[Callable],  # noqa: VULTURE
    *args,
    **kwargs,
):
    """Solve an ODE defined by the function arguments. Based on GOTM's *util/ode_solvers_template*.

    Args:
        ode_method (str): The numerical method ("FE": Forward Euler, "RK2": 2nd order Runge-Kutta,
            "RK4": 4th order Runge-Kutta)
        concentration (Union[TensorDict, torch.Tensor]): The current state
        dt (Union[float, torch.Tensor]): The time step
        dynamic_inputs (TensorDict): Inputs to the model, e.g. temperature, salinity, ...
        rhs_function (Callable): Right-hand side function
        ppdd_function (Optional[Callable]): Production and destruction matrices used for Patankar schemes
            (not yet implemented)

    Raises:
        NotImplementedError: The specified ODE solver has not been implemented yet
        ValueError: The specified ODE solver is invalid

    Returns:
        (TensorDict or torch.Tensor): The new state
    """
    if ode_method == "FE":
        return euler_forward(concentration, dt, dynamic_inputs, rhs_function, *args, **kwargs)
    elif ode_method == "RK2":
        return runge_kutta_2(concentration, dt, dynamic_inputs, rhs_function, *args, **kwargs)
    elif ode_method == "RK4":
        return runge_kutta_4(concentration, dt, dynamic_inputs, rhs_function, *args, **kwargs)
    elif (
        ode_method == "Patankar1"
        or ode_method == "Patankar2"
        or ode_method == "Patankar4"
        or ode_method == "MP1"
        or ode_method == "MP2"
        or ode_method == "MP4"
        or ode_method == "EMP1"
        or ode_method == "EMP2"
        or ode_method == "EMP4"
    ):
        raise NotImplementedError(f"The ODE solver {ode_method} has not been implemented yet")
    else:
        raise ValueError(f"{ode_method} is not a valid scheme")


def euler_forward(
    concentration: Union[TensorDict, torch.Tensor],
    dt: Union[float, torch.Tensor],
    dynamic_inputs: TensorDict,
    rhs_function: Callable,
    *args,
    **kwargs,
):
    """Simple Euler forward scheme (first order) according to:

    .. math::
        c^{n+1} = c^n + \\Delta t f(c^n)

    Args:
        concentration (Union[TensorDict, torch.Tensor]): The current state
        dt (Union[float, torch.Tensor]): The time step
        dynamic_inputs (TensorDict): Inputs to the model, e.g. temperature, salinity, ...
        rhs_function (Callable): Right-hand side function :math:`f`

    Returns:
        (TensorDict or torch.Tensor): The new state
    """
    rhs = rhs_function(concentration, dynamic_inputs, first=True, *args, **kwargs)

    concentration = partial_add_tensordict(concentration, dt * rhs)

    return concentration


def runge_kutta_2(
    concentration: Union[TensorDict, torch.Tensor],
    dt: Union[float, torch.Tensor],
    dynamic_inputs: TensorDict,
    rhs_function: Callable,
    *args,
    **kwargs,
):
    """Second-order explicit Runge-Kutta scheme, midpoint method according to:

    .. math::
        c^{n+1} = c^n + \\Delta t f(c^n + \\frac{\\Delta t}{2}f(c^n))

    Args:
        concentration (Union[TensorDict, torch.Tensor]): The current state
        dt (Union[float, torch.Tensor]): The time step
        dynamic_inputs (TensorDict): Inputs to the model, e.g. temperature, salinity, ...
        rhs_function (Callable): Right-hand side function :math:`f`

    Returns:
        (TensorDict or torch.Tensor): The new state
    """
    # Midpoint method

    rhs = rhs_function(concentration, dynamic_inputs, first=True, *args, **kwargs)

    concentration_1 = partial_add_tensordict(concentration, dt / 2.0 * rhs)

    rhs = rhs_function(concentration_1, dynamic_inputs, first=False, *args, **kwargs)

    concentration = partial_add_tensordict(concentration, dt * rhs)

    return concentration


def runge_kutta_4(
    concentration: Union[TensorDict, torch.Tensor],
    dt: Union[float, torch.Tensor],
    dynamic_inputs: TensorDict,
    rhs_function: Callable,
    *args,
    **kwargs,
):
    """Fourth-order explicit Runge-Kutta scheme according to:

    .. math::
        k_1 &= f(c^n) \\\\
        k_2 &= f(c^n + \\frac{\\Delta t}{2} k_1) \\\\
        k_3 &= f(c^n + \\frac{\\Delta t}{2} k_2) \\\\
        k_4 &= f(c^n + \\Delta t\, k_3) \\\\
        c^{n+1} &= c^n + \\frac{\\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)

    Args:
        concentration (Union[TensorDict, torch.Tensor]): The current state
        dt (Union[float, torch.Tensor]): The time step
        dynamic_inputs (TensorDict): Inputs to the model, e.g. temperature, salinity, ...
        rhs_function (Callable): Right-hand side function :math:`f`

    Returns:
        (TensorDict or torch.Tensor): The new state
    """
    rhs = rhs_function(concentration, dynamic_inputs, first=True, *args, **kwargs)

    rhs_1 = rhs / 2.0

    concentration_1 = partial_add_tensordict(concentration, dt * rhs_1)

    rhs = rhs_function(concentration_1, dynamic_inputs, first=False, *args, **kwargs)

    rhs_1 = rhs_1 + rhs

    concentration_1 = partial_add_tensordict(concentration, dt / 2 * rhs)

    rhs = rhs_function(concentration_1, dynamic_inputs, first=False, *args, **kwargs)

    rhs_1 = rhs_1 + rhs

    concentration_1 = partial_add_tensordict(concentration, dt * rhs)

    rhs = rhs_function(concentration_1, dynamic_inputs, first=False, *args, **kwargs)

    rhs_1 = rhs_1 + rhs / 2.0

    concentration = partial_add_tensordict(concentration, dt / 3.0 * rhs_1)

    return concentration


def partial_add_tensordict(a: TensorDict, b: TensorDict) -> TensorDict:
    """Helper function for adding TensorDicts a and b, where the keys of b are a subset of the keys of a.

    Args:
        a (TensorDict): One TensorDict
        b (TensorDict): Another TensorDict which has only keys also present in a

    Returns:
        TensorDict: a + b
    """
    if not isinstance(a, TensorDict):
        return a + b

    _increment = b.apply(torch.add, a)
    return a.copy().update(_increment)
