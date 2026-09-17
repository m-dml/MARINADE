from typing import Union

import torch
from tensordict import TensorDict


class BaseModel:
    """Base class for biosystem models.

    Subclasses should be defined for specific models.

    Instances of this base model will return zero sinking velocity and the forward step will not change the tracer
    concentrations.

    If created by the constructor method, this model uses the default parameters. A model with custom parameters can
    be created from a .yaml file with *model.create_model_from_yaml_file*.

    Args:
        model_name (str, optional): The name of the model. Defaults to "".
        repair_state (bool, optional): Whether to repair invalid states. Defaults to False.
        repair_minfunc (callable, optional): Method for the lower bound
            of allowed values. Defaults to torch.clamp_min.
        repair_maxfunc (callable, optional): Method for the upper bound
            of allowed values. Defaults to torch.clamp_max.
        device (str, optional): The device where the tensors of the model should be stored. Defaults to "cpu".
        dtype (torch.dtype, optional): The default data type of the model. Defaults to torch.float64.
    """

    def __init__(
        self,
        model_name: str = "",
        repair_state: bool = False,
        repair_minfunc: callable = torch.clamp_min,
        repair_maxfunc: callable = torch.clamp_max,
        mask_func: callable = lambda x, a, b: torch.where(x > 0, a, b),
        device="cpu",
        dtype=torch.float64,
        *args,
        **kwargs,
    ):
        self.model_name = model_name
        self.tracer_names = []  # the names of the tracers, e.g. "sed_c"
        self.benthic_tracer_names = []
        self.dependencies = []
        self.output_fields = []
        self.repair_state = repair_state
        self.repair_minfunc = repair_minfunc
        self.repair_maxfunc = repair_maxfunc
        self.parameters = TensorDict({}, device=device)

        self.mask_func = mask_func

    @classmethod
    def create_model_from_yaml_file(
        cls,
        path_fabm: str,
        path_gotm: str,
        device: str = "cpu",
        dtype: Union[str, torch.dtype] = torch.float64,
        *args,
        **kwargs,
    ):
        """Creates a *base_model* instance from .yaml configuration files. Should generally be used instead of the
        constructor. Designed to be compatible with the configuration files of GOTM.

        Should be overwritten by subclasses.

        Args:
            path_fabm (str): Path to the FABM .yaml configuration file
            path_gotm (str): Path to the GOTM .yaml configuration file
            device (str, optional): Torch device. Defaults to "cpu".
            dtype (str or torch.dtype, optional): Torch data type. Defaults to torch.float64.

        Returns:
            BaseModel: The generated model
        """
        raise NotImplementedError

    def get_vertical_movement(
        self, concentration: Union[TensorDict, torch.Tensor], dynamic_inputs: TensorDict, *args, **kwargs
    ) -> torch.Tensor:
        """Compute velocity for vertical movement *through* the fluid for biological state variables.

        Note that this method returns a zero sinking velocity. It should generally be overwritten
        by subclasses, in case they feature a non-zero sinking velocity.

        For more complex models, the result might depend on the momentary state.

        Args:
            concentration (Union[TensorDict, torch.Tensor]): Momentary concentrations of the model tracers.
                [batch, tracer, nlev]
            dynamic_inputs (TensorDict): Dict of additional inputs.

        Returns:
            torch.Tensor: Vertical movement of the model tracers.
                [batch, tracer or 1, nlev] or [batch, tracer or 1, nlev + 1]
        """
        if isinstance(concentration, torch.Tensor):
            x = concentration
        elif isinstance(concentration, TensorDict):
            x = concentration["pelagic"]
        shape = list(x.shape)
        shape[2] = shape[2] + 1
        return torch.zeros(shape, device=x.device, dtype=x.dtype)  # no sinking by default

    def do_repair_state(self, concentration: Union[TensorDict, torch.Tensor]):
        """Repair function for the states. In many models, there are maximal or minimal allowed values for certain
        tracers. In particular, there might be a positivity constraint.

        Note that this base implementation returns the current state without modifications.

        Args:
            concentration (Union[TensorDict, torch.Tensor]): Current state. Can be a tensor or TensorDict
                with key 'pelagic' for the submerged tracers.

        Returns:
            Union[TensorDict, torch.Tensor]: The repaired state
        """
        return concentration

    def forward(
        self, concentration: Union[TensorDict, torch.Tensor], dt: float, dynamic_inputs: TensorDict, *args, **kwargs
    ) -> Union[TensorDict, torch.Tensor]:
        """Compute the biogeochemical model time step based on the momentary state. Might require additional forcing
        inputs, e.g. incoming radiation, depending on the model.

        Args:
            concentration (Union[TensorDict, torch.Tensor]): The current tracer concentrations
            dt (float): The time step of the simulation
            dynamic_inputs (TensorDict): Dict of additional inputs

        Returns:
            (Union[TensorDict, torch.Tensor]): The new tracer concentration field
        """
        return concentration

    __call__ = forward

    def update_parameters(self, parameters: TensorDict):
        """Update the parameters of the model.

        Args:
            parameters (TensorDict): The model parameters to be updated
        """
        self.parameters.update(parameters)

    def get_parameters(self):
        """Retrieve the parameters of the model.

        Returns:
            (TensorDict): The model parameters
        """
        return self.parameters.copy()


NPZDModel = BaseModel  # legacy name
