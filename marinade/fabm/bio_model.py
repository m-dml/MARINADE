from abc import ABC
from typing import Union

import torch
from marinade.fabm.base_model import BaseModel
from marinade.util.ode_solvers import ode_solver
from tensordict import TensorDict


def shared_key_tensordicts(td1: TensorDict, td2: TensorDict):
    """Helper function.

    Finds intersection of TensorDict keys and returns corresponding subsets.
    """
    shared = set(td1.keys()) & set(td2.keys())  # set intersection of keys
    return td1.select(*shared), td2.select(*shared)


class BioModel(BaseModel, ABC):
    """Extension of the base class for biosystem models that implements crucial methods for modelling biogeochemical
    processes.

    Subclasses should be defined for specific models. They should implement the following methods:
    - *do*: Calculate derivative vector for pelagic tracer concentrations.
    - *do_surface*: Optional. Calculate surface derivatives for pelagic tracer concentrations.
    - *do_bottom*: Optional. Calculate bottom derivatives for pelagic tracer concentrations and derivatives for
        benthic fields.
    """

    def __init__(self, ode_method, device="cpu", dtype=torch.float64, *args, **kwargs):
        """Constructor method.

        If created by the constructor method, this model may use model-specific default parameters. A model with custom
        parameters can be created from a .yaml file with *model.create_model_from_yaml_file*, if implemented.
        """
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)

        super().__init__(device=device, dtype=dtype, *args, **kwargs)

        # constant velocity tensor for BioModel.get_vertical_movement
        self.ws = torch.tensor(0.0, device=device, dtype=dtype).unsqueeze(0).unsqueeze(-1)  # sinking velocity

        method_dict = {
            1: "FE",
            2: "RK2",
            3: "RK4",
            4: "Patankar1",
            5: "Patankar2",
            7: "MP1",
            8: "MP2",
            10: "EMP1",
            11: "EMP2",
        }
        if ode_method in method_dict:
            ode_method = method_dict[ode_method]

        self.ode_method = ode_method

    def do(
        self, concentration: Union[TensorDict, torch.Tensor], dynamic_inputs: TensorDict, *args, **kwargs
    ) -> Union[TensorDict, torch.Tensor]:
        """Calculate right-hand side of biological time step ODE as a vector with temporal derivatives.

        Returns:
            Union[TensorDict, torch.Tensor]: Vector of temporal derivatives
        """
        raise NotImplementedError

    def do_ppdd(
        self, concentration: Union[TensorDict, torch.Tensor], dynamic_inputs: TensorDict, *args, **kwargs
    ) -> tuple[Union[TensorDict, torch.Tensor], Union[TensorDict, torch.Tensor]]:
        """Calculate source/sink terms of biological variables as production and destruction matrices.

        Returns:
            Union[TensorDict, torch.Tensor]: production matrices
            Union[TensorDict, torch.Tensor]: destruction matrices
        """
        raise NotImplementedError

    def right_hand_side_rhs(
        self, concentration: Union[TensorDict, torch.Tensor], dynamic_inputs: TensorDict, first=True, *args, **kwargs
    ) -> Union[TensorDict, torch.Tensor]:
        """Calculate the time derivatives of the state. Calls self.do, self.do_surface and self.do_bottom, if
        implemented.

        Args:
            concentration (Union[TensorDict, torch.Tensor]): The current state.
            dynamic_inputs (TensorDict): Inputs to the model, e.g. temperature, salinity, ...
            first (bool, optional): If False, *self.do_repair_state* is called. Defaults to True.

        Returns:
            Union[TensorDict, torch.Tensor]: The time derivatives of the state
        """
        if not first:
            concentration = self.do_repair_state(concentration)
        rhs = self.do(concentration, dynamic_inputs, *args, **kwargs)
        if hasattr(self, "do_surface"):
            rhs_surface = self.do_surface(concentration, dynamic_inputs, *args, **kwargs)
            if isinstance(rhs, TensorDict):
                rhs["pelagic"] = rhs["pelagic"].index_add(
                    dim=-1,
                    index=torch.tensor(rhs["pelagic"].shape[-1] - 1, device=rhs["pelagic"].device),
                    source=rhs_surface["pelagic"] / dynamic_inputs["h"][..., -1:],
                )
            else:
                rhs = rhs.index_add(
                    dim=-1,
                    index=torch.tensor(rhs.shape[-1] - 1, device=rhs.device),
                    source=rhs_surface / dynamic_inputs["h"][..., -1:],
                )
        if hasattr(self, "do_bottom"):
            rhs_bottom = self.do_bottom(concentration, dynamic_inputs, *args, **kwargs)
            if isinstance(rhs, TensorDict):
                rhs["pelagic"] = rhs["pelagic"].index_add(
                    dim=-1,
                    index=torch.tensor(0, device=rhs["pelagic"].device),
                    source=rhs_bottom["pelagic"] / dynamic_inputs["h"][..., :1],
                )
                if "benthic" in rhs_bottom:
                    if "benthic" in rhs:
                        rhs["benthic"] = rhs["benthic"] + rhs_bottom["benthic"]
                    else:
                        rhs["benthic"] = rhs_bottom["benthic"]
            else:
                rhs = rhs.index_add(
                    dim=-1,
                    index=torch.tensor(0, device=rhs.device),
                    source=rhs_bottom / dynamic_inputs["h"][..., :1],
                )

        return rhs

    def right_hand_side_ppdd(
        self, concentration: Union[TensorDict, torch.Tensor], dynamic_inputs: TensorDict, *args, **kwargs
    ):
        """Calculate the source/sink terms of biological variables as production and destruction matrices.

        Not yet implemented.

        Args:
            concentration (_type_): The current state.
            dynamic_inputs (_type_): Inputs to the model, e.g. temperature, salinity, ...

        Returns:
            Union[TensorDict, torch.Tensor]: pp (production matrix)
            Union[TensorDict, torch.Tensor]: dd (destruction matrix)
        """
        return self.do_ppdd(concentration, dynamic_inputs, *args, **kwargs)

    def do_repair_state(self, concentration: Union[TensorDict, torch.Tensor]):
        """Repair function for the states. In many models, there are maximal or minimal allowed values for certain
        tracers. In particular, there might be a positivity constraint.

        Note that the variable *self.repair_state* (bool) indicates whether to do repairs.
        Only if True, this function applies corrections.

        The maximum and minimum allowed values should be specified by the model as *self.maximum*/*self.minimum*
        (for tensor states) or *self.maximum_td*/*self.minimum_td* (for TensorDict states).

        The correction functions should be stored as *self.repair_minfunc*
        or *self.repair_maxfunc*, e.g. *torch.clamp_min*.

        Args:
            concentration (Union[TensorDict, torch.Tensor]): Current state. Can be a tensor or TensorDict
                with key 'pelagic' for the submerged tracers.

        Returns:
            Union[TensorDict, torch.Tensor]: The repaired state
        """
        if self.repair_state:
            if isinstance(concentration, torch.Tensor):
                if hasattr(self, "minimum"):
                    concentration = self.repair_minfunc(concentration, self.minimum)
                if hasattr(self, "maximum"):
                    concentration = self.repair_maxfunc(concentration, self.maximum)
            elif isinstance(concentration, TensorDict):
                if hasattr(self, "minimum_td"):
                    concentration_shared, minimum_shared = shared_key_tensordicts(concentration, self.minimum_td)
                    concentration.update(concentration_shared.apply(self.repair_minfunc, minimum_shared))
                if hasattr(self, "maximum_td"):
                    concentration_shared, maximum_shared = shared_key_tensordicts(concentration, self.maximum_td)
                    concentration.update(concentration_shared.apply(self.repair_maxfunc, maximum_shared))
        return concentration

    def forward(
        self,
        concentration: Union[TensorDict, torch.Tensor],
        dt: Union[torch.Tensor, float],
        dynamic_inputs: TensorDict,
        *args,
        **kwargs,
    ) -> Union[TensorDict, torch.Tensor]:
        """Compute the biogeochemical model time step based on the momentary state. Might require additional forcing
        inputs, e.g. incoming radiation, depending on the model.

        Args:
            concentration (Union[TensorDict, torch.Tensor]): The current tracer concentrations
            dt (Union[torch.Tensor, float]): The time step of the simulation
            dynamic_inputs (TensorDict): Dict of additional inputs

        Returns:
            Union[TensorDict, torch.Tensor]: The new tracer concentration field
        """

        concentration = ode_solver(
            self.ode_method, concentration, dt, dynamic_inputs, self.right_hand_side_rhs, self.right_hand_side_ppdd
        )

        return concentration

    __call__ = forward

    def get_vertical_movement(
        self, concentration: Union[TensorDict, torch.Tensor], dynamic_inputs: TensorDict, *args, **kwargs
    ):
        """Compute vertical movement *through* the fluid for biological state variables.

        The default implementation for subclasses of BioModel returns a constant sinking velocity per tracer.
        It should be stored during model initialization as self.ws with the shape [1, tracer, 1].

        Args:
            concentration (Union[TensorDict, torch.Tensor]): Momentary concentrations of the model tracers.
                [batch, tracer, nlev]
            dynamic_inputs (TensorDict): Dict of additional inputs.

        Returns:
            torch.Tensor: Constant vertical velocity as defined at model initialization.
                [batch, tracer or 1, nlev]
        """
        self.ws = self.ws.to(device=concentration.device)
        return self.ws
