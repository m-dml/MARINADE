from typing import Optional, Union

import torch
from marinade.gotm.tridiagonal import tridiagonal_solve
from tensordict import TensorDict
from torch import tensor


def diffuse_at_center(
    value: torch.Tensor,
    dt: float,
    h: torch.Tensor,
    nu: torch.Tensor,
    bc_up_type: str,
    bc_down_type: str,
    bc_up_value: Union[float, torch.Tensor],
    bc_down_value: Union[float, torch.Tensor],
    implicitness: float = 1.0,
    allow_negative: bool = False,
    lsource: Optional[torch.Tensor] = None,
    qsource: Optional[torch.Tensor] = None,
    method: str = "thomas",
    implicit_backward: bool = True,
) -> torch.Tensor:
    r"""Diffuse a tracer at layer centers using GOTM diffusion operator.

    Solves the one-dimensional diffusion equation:

    .. math::
        \frac{\partial Y}{\partial t} = \frac{\partial}{\partial z}\left(\nu_Y\frac{\partial Y}{\partial z}\right)
        + Y L_{sour} + Q_{sour}

    Args:
        value (torch.Tensor): Initial tracer concentration. [batch, tracer, nlev]
        dt (float): Time step.
        h (torch.Tensor): Layer heights. [batch, 1, nlev]
        nu (torch.Tensor): Turbulent diffusivities. [batch, 1, nlev+1]
        bc_up_type (str): Type of the upper boundary condition. "neumann" or "dirichlet".
        bc_down_type (str): Type of the lower boundary condition. "neumann" or "dirichlet".
        bc_up_value (Union[float, torch.Tensor]): Value of the upper boundary condition.
            Scalar or [batch, (tracer or 1)]
        bc_down_value (Union[float, torch.Tensor]): Value of the lower boundary condition.
            Scalar or [batch, (tracer or 1)]
        implicitness (float, optional): The implicitness of the numerical scheme. Defaults to 1.0.
        allow_negative (bool, optional): Whether to allow negative values of the tracer. Defaults to False.
        lsource (torch.Tensor, optional): Linear source term. Defaults to None. [batch, (tracer or 1), nlev]
        qsource (torch.Tensor, optional): Constant source term. Defaults to None. [batch, (tracer or 1), nlev]
        method (str, optional): Method for tridiagonal solver. One of ("thomas", "paracr", "linalg").
            Defaults to "thomas".
        implicit_backward (bool, optional): If True, uses custom backward method based on implicit differentiation.
            Defaults to True.

    Returns:
        torch.Tensor: Final tracer concentration, shape (batch, nlev)
    """

    if isinstance(bc_up_value, float):
        bc_up_value = torch.tensor(bc_up_value, dtype=value.dtype, device=value.device)
    if isinstance(bc_down_value, float):
        bc_down_value = torch.tensor(bc_down_value, dtype=value.dtype, device=value.device)

    au, bu, cu, yu = setup_tridiagonal_matrix(
        value,
        dt,
        h,
        nu,
        bc_up_type,
        bc_down_type,
        bc_up_value,
        bc_down_value,
        implicitness=implicitness,
        allow_negative=allow_negative,
        lsource=lsource,
        qsource=qsource,
    )
    solution = tridiagonal_solve(au, bu, cu, yu, method, implicit_backward)
    return solution


# @torch.jit.script
def setup_boundary(
    dt: torch.Tensor,
    implicitness: float,
    allow_negative: bool,
    h_boundary: torch.Tensor,
    h_adjacent: torch.Tensor,
    nu_boundary: torch.Tensor,
    value_boundary: torch.Tensor,
    value_adjacent: torch.Tensor,
    ls: torch.Tensor,
    qs: torch.Tensor,
    bc_type: str,
    bc_value: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
):
    """Sets up a row of a tridiagonal system for a boundary condition."""
    if bc_type in ("flux", "neumann"):
        v = 2 * dt * nu_boundary / (h_boundary + h_adjacent) / h_boundary
        offdiag = -implicitness * v  # a or c
        if not allow_negative and (bc_value < 0.0).any():  # Patankar (1980) trick
            _one_minus = 1 - offdiag
            _div = torch.where(bc_value < 0.0, value_boundary * h_boundary, 1.0)
            # prevent NaN gradients: don't divide by zero
            _where_zero = _div == 0.0
            _div = torch.where(_where_zero, 1.0, _div)
            _val = torch.where(bc_value < 0.0, _one_minus - dt * bc_value / _div, _one_minus)
            _val = torch.where(_where_zero, torch.inf, _val)
            # torch.inf is ok because later on, only 1/_val is used. This will give zero.
            _val1 = torch.where(bc_value < 0.0, torch.tensor(0.0, dtype=dtype, device=device), bc_value / h_boundary)
        else:
            _val = 1 - offdiag
            _val1 = bc_value / h_boundary
        b = _val - ls
        y = value_boundary + dt * _val1 + (1 - implicitness) * v * (value_adjacent - value_boundary) + qs
    elif bc_type in ("value", "dirichlet"):
        b = torch.ones_like(h_boundary)
        offdiag = torch.zeros_like(h_boundary)
        y = bc_value.expand_as(value_boundary)
    else:
        raise ValueError(f"Invalid boundary condition type: {bc_type}")
    return b, offdiag, y


# @torch.jit.script
def setup_tridiagonal_matrix(
    value: torch.Tensor,
    dt: torch.Tensor,
    h: torch.Tensor,
    nu: torch.Tensor,
    bc_up_type: str,
    bc_down_type: str,
    bc_up_value: torch.Tensor,
    bc_down_value: torch.Tensor,
    implicitness: float = 1.0,
    allow_negative: bool = False,
    lsource: Optional[torch.Tensor] = None,
    qsource: Optional[torch.Tensor] = None,
):
    """Set up a batch of tridiagonal matrices for GOTM diffusion operator.

    Args:
        value (torch.Tensor): Current state of tracer concentrations. [batch, tracer, nlev]
        dt (float): Time step
        h (torch.Tensor): Layer thickness. [batch, 1, nlev]
        nu (torch.Tensor): Turbulent coefficients. [batch, 1, nlev+1]
        bc_up_type (str): Type of upper boundary condition. Options: 'neumann', 'dirichlet'
        bc_down_type (str): Type of lower boundary condition. Options: 'neumann', 'dirichlet'
        bc_up_value (torch.Tensor): Value of upper boundary condition. Scalar or [batch, (tracer or 1)]
        bc_down_value (torch.Tensor): Value of lower boundary condition. Scalar or [batch, (tracer or 1)]
        implicitness (float, optional): Implicitness parameter. Defaults to 1.0.
        allow_negative (bool, optional): Allow negative values. Defaults to False.
        lsource (torch.Tensor, optional): Linear source term. Defaults to None. [batch, (tracer or 1), nlev]
        qsource (torch.Tensor, optional): Constant source term. Defaults to None. [batch, (tracer or 1), nlev]

    Returns:
        Tuple:

        - a (torch.Tensor): lower diagonal of shape (batch, 1, nlev - 1)
        - b (torch.Tensor): main diagonal of shape (batch, (tracer or 1), nlev)
        - c (torch.Tensor): upper diagonal of shape (batch, 1, nlev - 1)
        - y (torch.Tensor): right-hand side of shape (batch, tracer, nlev)
    """

    dtype = nu.dtype
    device = nu.device

    batch_size, n_tracers, n_levels = value.shape

    au = torch.empty((batch_size, nu.shape[-2], n_levels - 1), device=device, dtype=dtype)

    # do we need a different diagonal coefficient for each tracer? first check if lsource varies over tracers:
    bu_dim2 = 1 if (lsource is None or lsource.shape[-2] == 1) and nu.shape[-2] == 1 else n_tracers
    # if we need to enforce nonnegativity constraints, then b depends on values and must have a tracer dimension:
    for bc_type, bc_val in zip([bc_up_type, bc_down_type], [bc_up_value, bc_down_value]):
        if bc_type in ("flux", "neumann") and not allow_negative and (bc_val < 0.0).any():
            bu_dim2 = n_tracers
    bu = torch.empty(value.shape[:-2] + (bu_dim2, n_levels), device=device, dtype=dtype)

    cu = torch.empty((batch_size, nu.shape[-2], n_levels - 1), device=device, dtype=dtype)
    yu = torch.empty_like(value)

    # Set up the linear system except for boundaries:
    c = 2 * dt * nu[..., 2:n_levels] / (h[..., 1 : n_levels - 1] + h[..., 2:n_levels]) / h[..., 1 : n_levels - 1]
    a = (
        2
        * dt
        * nu[..., 1 : n_levels - 1]
        / (h[..., 1 : n_levels - 1] + h[..., : n_levels - 2])
        / h[..., 1 : n_levels - 1]
    )

    cu[..., 1 : n_levels - 1] = -implicitness * c
    au[..., 0 : n_levels - 2] = -implicitness * a

    ls = tensor(0.0, device=device, dtype=dtype) if lsource is None else dt * lsource[..., 1 : n_levels - 1]
    bu[..., 1 : n_levels - 1] = 1 + implicitness * (a + c) - ls

    qs = tensor(0.0, device=device, dtype=dtype) if qsource is None else dt * qsource[..., 1 : n_levels - 1]
    yu[..., 1 : n_levels - 1] = (
        (1 - (1 - implicitness) * (a + c)) * value[..., 1 : n_levels - 1]
        + (1 - implicitness) * (a * value[..., : n_levels - 2] + c * value[..., 2:n_levels])
        + qs
    )

    # set up the first and last rows, which depend on boundary conditions
    for i_bc, i_adj, bc_type, bc_val, offdiag in zip(
        [-1, 0], [-2, 1], [bc_up_type, bc_down_type], [bc_up_value, bc_down_value], [au, cu]
    ):
        bu[..., i_bc], offdiag[..., i_bc], yu[..., i_bc] = setup_boundary(
            dt,
            implicitness,
            allow_negative,
            h[..., i_bc],
            h[..., i_adj],
            nu[..., i_adj],
            value[..., i_bc],
            value[..., i_adj],
            (tensor(0.0, device=device, dtype=dtype) if lsource is None else dt * lsource[..., i_bc]),
            (tensor(0.0, device=device, dtype=dtype) if qsource is None else dt * qsource[..., i_bc]),
            bc_type,
            bc_val,
            dtype,
            device,
        )

    return au, bu, cu, yu


class DiffusionOperator:
    """Diffusion operator class.

    Should be initialized with the important numerical parameters and schemes for the simulation. Has a *forward* method
    for computing a single time step. Calling the operator object itself also calls its *forward* method.

    The operator solves the one-dimensional diffusion equation:

    .. math::
        \\frac{\partial Y}{\partial t} = \\frac{\partial}{\partial z}\left(\\nu_Y\\frac{\partial Y}{\partial z}\\right)
        + Y L_{sour} + Q_{sour}

    Args:
        bc_up_type (str): Type of upper boundary condition. "neumann" or "dirichlet".
        bc_down_type (str): Type of lower boundary condition. "neumann" or "dirichlet".
        bc_up_value (Union[float, torch.Tensor]): Value of upper boundary condition. Scalar or [batch, (tracer or 1)]
        bc_down_value (Union[float, torch.Tensor]): Value of lower boundary condition. Scalar or [batch, (tracer or 1)]
        implicitness (float, optional): Implicitness parameter. Defaults to 1.0.
        allow_negative (bool, optional): Allow negative tracer concentration values. Defaults to False.
        method (str, optional): Method for tridiagonal solver. One of ("thomas", "paracr", "linalg").
            Defaults to "thomas".
        implicit_backward (bool, optional): If True, uses custom backward method based on implicit differentiation.
            Defaults to True.
        device (str, optional): Device for torch computations. Defaults to 'cpu'.
        dtype (str, optional): Data type for torch computations. Defaults to 'float64'.
    """

    def __init__(
        self,
        bc_up_type: str,
        bc_down_type: str,
        bc_up_value: Union[float, torch.Tensor],
        bc_down_value: Union[float, torch.Tensor],
        implicitness: float = 1.0,
        allow_negative: bool = False,
        method: str = "thomas",
        implicit_backward: bool = True,
        device: str = "cpu",
        dtype="float64",
        *args,
        **kwargs,
    ):
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.implicitness = implicitness
        self.allow_negative = allow_negative
        self.bc_up_type = bc_up_type
        self.bc_down_type = bc_down_type
        if isinstance(bc_up_value, int):
            bc_up_value = float(bc_up_value)
        self.bc_up_value = bc_up_value
        if isinstance(bc_down_value, int):
            bc_down_value = float(bc_down_value)
        self.bc_down_value = bc_down_value
        self.method = method
        self.implicit_backward = implicit_backward

    def forward(
        self,
        concentration: Union[torch.Tensor, TensorDict],
        dt: float,
        h: torch.Tensor,
        nu: torch.Tensor,
        lsource: Optional[torch.Tensor] = None,
        qsource: Optional[torch.Tensor] = None,
        bc_up_value: Optional[Union[float, torch.Tensor]] = None,
        bc_down_value: Optional[Union[float, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Forward function for time stepping operator.

        Args:
            concentration (Union[torch.Tensor, TensorDict]): The initial tracer concentration field.
                Can be a TensorDict. Then this operator acts on *concentration["pelagic"]*.
                This functionality exists because benthic (sediment) variables might be part
                of the tracer state, but they are only on the ground and do not diffuse.
                [batch, tracer, nlev]
            dt (float): Time step.
            h (torch.Tensor): Layer heights. [batch, (tracer or 1), nlev]
            nu (torch.Tensor): Diffusivity field at cell faces. [batch, (tracer or 1), nlev+1]
            lsource (torch.Tensor, optional): Linear source term. Defaults to None.
            qsource (torch.Tensor, optional): Constant source term. Defaults to None.
            bc_up_value (Union[float, torch.Tensor], optional): Upper boundary condition value. Optional.
                If None, the value specified at operator initialization is used. Defaults to None.
                Scalar or [batch, (tracer or 1)]
            bc_down_value (Union[float, torch.Tensor], optional): Lower boundary condition value. Optional.
                If None, the value specified at operator initialization is used. Defaults to None.
                Scalar or [batch, (tracer or 1)]

        Returns:
            Union[torch.Tensor, TensorDict]: Updated system state.
        """
        if isinstance(concentration, torch.Tensor):
            x = concentration
        elif isinstance(concentration, TensorDict):
            x = concentration["pelagic"]
        x_new = diffuse_at_center(
            x,
            dt,
            h,
            nu,
            self.bc_up_type,
            self.bc_down_type,
            self.bc_up_value if bc_up_value is None else bc_up_value,
            self.bc_down_value if bc_down_value is None else bc_down_value,
            self.implicitness,
            self.allow_negative,
            lsource,
            qsource,
            self.method,
            self.implicit_backward,
        )

        if isinstance(concentration, TensorDict):
            concentration = concentration.copy()
            concentration["pelagic"] = x_new
            return concentration
        else:
            return x_new

    __call__ = forward  # define alias for simplified calling
