from typing import Union

import torch
from tensordict import TensorDict


def adv_reconstruct(
    scheme: str, cfl: torch.Tensor, fuu: torch.Tensor, fu: torch.Tensor, fd: torch.Tensor
) -> torch.Tensor:
    r"""Reconstruction of the advected quantity at the cell interface using different schemes.

    This is a reimplementation of the Fortran subroutine ADV_RECONSTRUCT from GOTM.

    Computes the upstream-biased fluxes:

    .. math::
        F_i^n = \frac{1}{\Delta t}\int_{z_i^{\text{Face}} - w \Delta t}^{z_i^{\text{Face}}} Y(z') \,\text{dz'}

    Args:
        scheme (str): The name of the scheme ("superbee", "p2_pdm", "splmax13", "muscl", "central", "p2").
        cfl (torch.Tensor): The Courant number. [batch, (tracer or 1), nlev]
        fuu (torch.Tensor): The value at the upwind cell. [batch, tracer, nlev]
        fu (torch.Tensor): The value at the current cell. [batch, tracer, nlev]
        fd (torch.Tensor): The value at the downwind cell. [batch, tracer, nlev]

    Returns:
        torch.Tensor: Computed fluxes.
    """
    delta_f = fd - fu
    delta_fu = fu - fuu

    ratio = delta_fu / torch.where(delta_f != 0, delta_f, 1)  # important for preventing nan values in the gradients!

    if scheme == "superbee":
        limiter = torch.maximum(torch.clamp_max(2 * ratio, 1), torch.clamp_max(ratio, 2))  # torch.clamp_max = minimum
        alt_output = fu  # output if delta_f * delta_fu <= 0
    elif scheme == "p2_pdm":
        x = 1 / 6 * (1 - 2 * cfl)
        limiter = torch.minimum((0.5 + x) + (0.5 - x) * ratio, 2 / (1 - cfl))
        limiter = torch.minimum(2 * ratio / (cfl + 1e-10), limiter)
        alt_output = fu  # output if delta_f * delta_fu <= 0
    elif scheme == "splmax13":
        limiter = torch.minimum(torch.clamp_max(2 * ratio, 2), 1 / 3 * torch.clamp_min(1 + 2 * ratio, 2 + ratio))
        alt_output = fu  # output if delta_f * delta_fu <= 0
    elif scheme == "muscl":
        limiter = torch.minimum(2 * ratio, torch.clamp_max(0.5 * (1 + ratio), 2))
        alt_output = fu  # output if delta_f * delta_fu <= 0
    elif scheme == "p2":
        x = 1 / 6 * (1 - 2 * cfl)
        limiter = (0.5 + x) + (0.5 - x) * ratio
        alt_output = fu + 0.5 * (1 - cfl) * ((0.5 + x) * delta_f + (0.5 - x) * delta_fu)
    elif scheme == "central":
        limiter = 1 / (1 - cfl)
        alt_output = 0.5 * (fu + fd)
    else:  # scheme == "upstream"
        limiter = torch.zeros_like(cfl)
        alt_output = fu  # output if delta_f * delta_fu <= 0
    return torch.where(delta_f * delta_fu > 0, fu + 0.5 * limiter * (1 - cfl) * delta_f, alt_output)


# @torch.jit.script
def advect_at_center(
    value: torch.Tensor,
    dt: torch.Tensor,
    h: torch.Tensor,
    ww: torch.Tensor,
    bc_up_type: str,
    bc_down_type: str,
    bc_up_value: Union[torch.Tensor, int, float],
    bc_down_value: Union[torch.Tensor, int, float],
    scheme: str,
    conserve: bool,
) -> torch.Tensor:
    r"""Vertical advection of a quantity at the center of the grid cells. This is a reimplementation of the Fortran
    subroutine ADVECT_AT_CENTER from GOTM.

    If _conserve_ == _True_, the 1d conservative advection equation is used:

    .. math::
        \frac{\partial Y}{\partial t} = -\frac{\partial F}{\partial z}

    If _conserve_ == _False_, the 1d non-conservative advection equation is used:

    .. math::
        \frac{\partial Y}{\partial t} = -(\frac{\partial F}{\partial z} - Y \frac{\partial w}{\partial z}),

    where :math:`F` is the flux field due to the velocity :math:`w`.


    Args:
        value (torch.Tensor): Values of the advected quantity at cell centers. [n_batch, tracer, nlev]
        dt (torch.Tensor): Time step.
        h (torch.Tensor): Cell thicknesses. [n_batch, 1, nlev]
        ww (torch.Tensor): Vertical velocities at cell interfaces. [n_batch, (tracer or 1), nlev+1]
        bc_up_type (str): Type of upper boundary condition ("flux", "value", "oneSided", "zeroDivergence").
        bc_down_type (str): Type of lower boundary condition ("flux", "value", "oneSided", "zeroDivergence").
        bc_up_value (Union[torch.Tensor, int, float]): Value for upper boundary condition. Scalar or [batch, tracer]
        bc_down_value (Union[torch.Tensor, int, float]): Value for lower boundary condition. Scalar or [batch, tracer]
        scheme (str): Advection scheme to use ("superbee", "p2_pdm", "splmax13", "muscl", "central", "p2").
        conserve (bool): Whether to use the conservative form of the advection equation.

    Returns:
        torch.Tensor: Advected tracers.
    """

    device = value.device
    dtype = value.dtype
    nlev = value.shape[-1]
    if isinstance(bc_up_value, int):
        bc_up_value = torch.tensor(bc_up_value, device=value.device, dtype=value.dtype)
    elif isinstance(bc_up_value, float):
        bc_up_value = torch.tensor(bc_up_value, device=value.device, dtype=value.dtype)
    if isinstance(bc_down_value, int):
        bc_down_value = torch.tensor(bc_down_value, device=value.device, dtype=value.dtype)
    elif isinstance(bc_down_value, float):
        bc_down_value = torch.tensor(bc_down_value, device=value.device, dtype=value.dtype)

    itmax = 100

    cmax, _ = torch.max(torch.abs(ww[..., 1:nlev]) * (2 * dt) / (h[..., : nlev - 1] + h[..., 1:]), dim=-1, keepdim=True)
    it = (cmax + 1).clamp_max(itmax).to(torch.int)

    c = torch.abs(ww[..., 1:nlev]) * (2 * dt / it) / (h[..., : nlev - 1] + h[..., 1:])
    for i in range(1, int(torch.max(it).item()) + 1):  # time iteration
        Yu = torch.empty_like(value[..., 1:])
        Yu[..., 1:-1] = torch.where(ww[..., 2 : nlev - 1] > 0, value[..., :-3], value[..., 3:])
        Yu[..., 0] = torch.where(ww[..., 1] > 0, value[..., 0], value[..., 2])
        Yu[..., -1] = torch.where(ww[..., nlev - 1] > 0, value[..., -3], value[..., -1])
        Yc = torch.where(ww[..., 1:nlev] > 0, value[..., : nlev - 1], value[..., 1:])
        Yd = torch.where(ww[..., 1:nlev] > 0, value[..., 1:], value[..., : nlev - 1])

        cu = torch.empty([value.shape[-3], value.shape[-2], value.shape[-1] + 1], device=device, dtype=dtype)
        cu[..., 1:-1] = ww[..., 1:nlev] * adv_reconstruct(scheme, c, Yu, Yc, Yd)

        # Upper boundary conditions
        if bc_up_type == "flux":
            cu[..., nlev] = -bc_up_value
        elif bc_up_type == "value":
            cu[..., nlev] = ww[..., nlev] * bc_up_value
        elif bc_up_type == "oneSided":
            cu[..., nlev] = torch.where(ww[..., nlev] >= 0, ww[..., nlev] * value[..., nlev - 1], 0.0)
        elif bc_up_type == "zeroDivergence":
            cu[..., nlev] = cu[..., nlev - 1]
        else:
            raise ValueError("Unknown upper boundary condition type in adv_center()")

        # Lower boundary conditions
        if bc_down_type == "flux":
            cu[..., 0] = bc_down_value
        elif bc_down_type == "value":
            cu[..., 0] = ww[..., 0] * bc_down_value
        elif bc_down_type == "oneSided":
            cu[..., 0] = torch.where(ww[..., 0] > 0, 0.0, ww[..., 0] * value[..., 0])
        elif bc_down_type == "zeroDivergence":
            cu[..., 0] = cu[..., 1]
        else:
            raise ValueError("Unknown lower boundary condition type in adv_center()")

        # Vertical advection step
        if conserve:
            # Only update the value for each batch element as often as necessary.
            # The number of substeps per batch element should not depend on the
            # (potentially random) combination of batch elements.
            value = torch.where(i <= it, value - dt / it * ((cu[..., 1:] - cu[..., :nlev]) / h), value)
        else:
            value = torch.where(
                i <= it,
                value - dt / it * ((cu[..., 1:] - cu[..., :nlev]) - value * (ww[..., 1:] - ww[..., :nlev])) / h,
                value,
            )
    return value


class AdvectionOperator:
    """Advection operator class.

    Should be initialized with the important numerical parameters and schemes for the simulation. Has a *forward* method
    for computing a single time step. Calling the operator object itself also calls its *forward* method.

    The operator solves the conservative or non-conservative advection equation:

    .. math::
        \\frac{\partial Y}{\partial t} &= -\\frac{\partial F}{\partial z}
        \qquad && \\text{(Conservative form)} \\\\
        \\frac{\partial Y}{\partial t}
        &= -\left(\\frac{\partial F}{\partial z} - Y \\frac{\partial w}{\partial z}\\right)
        \qquad && \\text{(Non-conservative form)}

    :math:`F` is the flux field due to the velocity :math:`w`. It is computed in an upstream-biased way:

    .. math::
        F_i^n = \\frac{1}{\\Delta t}\\int_{z_i^{\\text{Face}} - w \\Delta t}^{z_i^{\\text{Face}}} Y(z') \\,\\text{dz'}

    Args:
        bc_up_type (str): Type of upper boundary condition ("flux", "value", "oneSided", "zeroDivergence").
        bc_down_type (str): Type of lower boundary condition ("flux", "value", "oneSided", "zeroDivergence").
        bc_up_value (Union[torch.Tensor, float, int]): Value for upper boundary condition. Scalar or [batch, tracer]
        bc_down_value (Union[torch.Tensor, float, int]): Value for lower boundary condition. Scalar or [batch, tracer]
        scheme (str): Advection limiter scheme to use ("superbee", "p2_pdm", "splmax13", "muscl", "central", "p2").
        conserve (bool): Whether to use the conservative form of the advection equation.
    """

    def __init__(
        self,
        bc_up_type: str,
        bc_down_type: str,
        bc_up_value: Union[torch.Tensor, float, int],
        bc_down_value: Union[torch.Tensor, float, int],
        scheme: str,
        conserve: bool,
        *args,
        **kwargs,
    ):
        if bc_up_type in ("flux", "value", "oneSided", "zeroDivergence"):
            self.bc_up_type = bc_up_type
        else:
            raise ValueError("Unknown upper boundary condition type")
        if bc_down_type in ("flux", "value", "oneSided", "zeroDivergence"):
            self.bc_down_type = bc_down_type
        else:
            raise ValueError("Unknown lower boundary condition type")
        self.bc_up_value = bc_up_value
        self.bc_down_value = bc_down_value
        if scheme in ("superbee", "p2_pdm", "splmax13", "muscl", "central", "p2"):
            self.scheme = scheme
        else:
            raise ValueError("Unknown vertical advection scheme")
        self.conserve = conserve

    def forward(
        self,
        concentration: Union[torch.Tensor, TensorDict],
        dt: float,
        h: torch.Tensor,
        w: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the state at the next time step from the current state after advection.

        Args:
            concentration (Union[torch.Tensor, TensorDict]): The initial tracer concentration field.
                Can be a TensorDict. Then this operator acts on *concentration["pelagic"]*.
                This functionality exists because benthic (sediment) variables might be part
                of the tracer state, but they are only on the ground and do not advect.
                [batch, tracer, nlev]
            dt (float): The time step.
            h (torch.Tensor): The layer heights. [batch, 1, nlev]
            w (torch.Tensor): The velocity field. This can be:

                - [batch, (tracer or 1), nlev]:   Values at cell centers
                - [batch, (tracer or 1), nlev+1]: Values at cell faces.
                - [batch, (tracer or 1), 1]: Constant velocity field (also at domain boundaries).

        Returns:
            torch.Tensor: The state at the next time step. [batch, tracer, nlev]
        """
        if isinstance(concentration, torch.Tensor):
            x = concentration
        elif isinstance(concentration, TensorDict):
            x = concentration["pelagic"]

        # w should be given at the cell faces
        if w.shape[-1] == x.shape[-1]:
            # w is given at the cell centers
            # Interpolate it at the faces
            iweights = h[..., 1:] / (h[..., :-1] + h[..., 1:])
            _shape = list(w.shape)
            _shape[-1] = x.shape[-1] + 1
            ws1d = torch.empty(_shape, device=w.device, dtype=w.dtype)
            ws1d[..., 1:-1] = iweights * w[..., :-1] + (1 - iweights) * w[..., 1:]
            ws1d[..., (0, -1)] = w[..., (0, -1)]
            w = ws1d
        elif w.shape[-1] == 1:
            # w is given as a constant
            _shape = list(w.shape)
            _shape[-1] = x.shape[-1] + 1
            ws1d = torch.empty(_shape, device=w.device, dtype=w.dtype)
            ws1d[..., :] = w
            w = ws1d
        elif w.shape[-1] != x.shape[-1] + 1:
            raise ValueError(
                "w does not have the expected shape! It should be of shape "
                "[batch, (tracer or 1), (nlev+1 or nlev or 1)]."
            )

        x_new = advect_at_center(
            x,
            dt,
            h,
            w,
            self.bc_up_type,
            self.bc_down_type,
            self.bc_up_value,
            self.bc_down_value,
            self.scheme,
            self.conserve,
        )

        if isinstance(concentration, TensorDict):
            concentration = concentration.copy()
            concentration["pelagic"] = x_new
            return concentration
        else:
            return x_new

    __call__ = forward  # define alias for simplified calling
