from typing import Union

import torch
import yaml
from marinade.fabm.base_model import BaseModel
from tensordict import TensorDict


class PassiveModel(BaseModel):
    """This class provides a very simple biosystem model for an inert tracer.

    There are no biogeochemical reactions, the tracer stays conserved and only sinks relative to the water with a
    constant velocity. The sinking velocity is the one model parameter of this class.
    """

    def __init__(
        self,
        ws: Union[torch.Tensor, float] = 0.0,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        *args,
        **kwargs,
    ):
        """Constructor method.

        Args:
            ws (Union[torch.Tensor, float]): The sinking velocity.
                Can be scalar or [(batch or 1), (tracer or 1), (nlev or 1)].
                Only use tensor when you are sure the dimensions will be compatible with the number of tracers in the
                simulation.
            device (str, optional): The device where the tensors of the model should be stored. Defaults to "cpu".
            dtype (torch.dtype, optional): The default data type of the model. Defaults to torch.float64.
        """
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        super().__init__(device=device, dtype=dtype, *args, **kwargs)
        if not isinstance(ws, torch.Tensor):
            ws = torch.tensor(ws, device=device, dtype=dtype)
        while ws.ndim < 3:
            ws = ws.unsqueeze(-1)

        self.parameters["ws"] = ws

        self.minimum = torch.tensor([0.0], device=device, dtype=dtype).unsqueeze(0).unsqueeze(-1)
        self.minimum_td = TensorDict({"pelagic": self.minimum})

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
        """Creates a passive_model instance based on .yaml configuration files.

        For this model, the required keys in the gotm yaml file are:
        - "grid/nlev": The number of vertical layers.
        - "instances/<instance_name>/model": Should be "bb/passive" for the passive model
        - "instances/<instance_name>/parameters/vertical_velocity": Sinking velocity
        - "instances/<instance_name>/initialization/<tracer_name>": Initial (homogeneous) concentrations for the
          tracers. The values are not actually used, but the tracer names are stored in *model.tracer_names* and
          the velocity field is initialized with the tracer dimension equal to the number of tracers.

        Args:
            path_fabm (str): Path to the FABM .yaml configuration file
            path_gotm (str): Path to the GOTM .yaml configuration file
            device (str, optional): Torch device. Defaults to "cpu".
            dtype (Union[str, torch.dtype]): Torch data type. Defaults to torch.float64.

        Returns:
            passive_model: The generated model
        """
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        model = cls(torch.tensor(0, device=device, dtype=dtype), device=device, dtype=dtype, *args, **kwargs)

        # load the GOTM and FABM .yaml config files:
        with open(path_gotm, "r") as file_gotm:
            config_gotm = yaml.safe_load(file_gotm)
        with open(path_fabm, "r") as file_fabm:
            config_fabm = yaml.safe_load(file_fabm)

        nlev = config_gotm["grid"]["nlev"]

        w_list = []

        for name_instance, config_instance in config_fabm["instances"].items():
            if config_instance["model"] == "bb/passive":
                try:
                    w_instance = config_instance["parameters"]["vertical_velocity"]
                except (KeyError, ValueError, TypeError):
                    raise ValueError(
                        f"instances/{name_instance}/parameters/vertical_velocity missing or invalid\
                                     in {path_fabm}.\
                                     Expected type float or int. The suggested default value is 0."
                    )
                try:
                    for name_tracer in config_instance["initialization"]:  # the model tracers
                        model.tracer_names.append(f"{name_instance}/{name_tracer}")
                        w_list.append(w_instance / (3600 * 24))  # the model velocity
                except (KeyError, ValueError, TypeError):
                    raise ValueError(
                        f"instances/{name_instance}/initialization/{name_tracer} missing or invalid\
                                     in {path_fabm}.\
                                     Expected type float or int. The suggested default value is 0."
                    )
        if len(set(w_list)) <= 1:  # only one velocity
            model.parameters["ws"] = torch.full((1, nlev + 1), w_list[0], device=device, dtype=dtype)
        else:
            model.parameters["ws"] = torch.tensor(w_list, device=device, dtype=dtype)
            model.parameters["ws"] = model.parameters["ws"].repeat(1, nlev + 1, 1)
        model.parameters["ws"][:, [0, -1]] = 0

        return model

    def get_vertical_movement(
        self, concentration: Union[torch.Tensor, TensorDict], dynamic_inputs, *args, **kwargs
    ) -> torch.Tensor:
        """Compute vertical movement relative to the fluid for biological state variables.

        For this model, returns a constant sinking velocity.

        Args:
            concentration (Union[torch.Tensor, TensorDict]): Momentary concentrations of the
                model tracers. [batch, tracer, nlev]
            dynamic_inputs (TensorDict): Dict of additional inputs.

        Returns:
            torch.Tensor: Constant vertical velocity as defined at model initialization.
                [batch, (tracer or 1), nlev]
        """
        if isinstance(concentration, torch.Tensor):
            x = concentration
        elif isinstance(concentration, TensorDict):
            x = concentration["pelagic"]

        self.parameters["ws"] = self.parameters["ws"].to(x)
        if self.parameters["ws"].shape[0] == 1:
            return self.parameters["ws"].expand(x.shape[0], *[-1] * (x.ndim - 1))
        return self.parameters["ws"]
