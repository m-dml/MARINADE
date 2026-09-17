from typing import List, Optional, Union

import torch
import yaml
from marinade.fabm.base_model import BaseModel
from marinade.util.temporal_mean import TemporalMean
from tensordict import TensorDict


class GOTMLightModel(BaseModel):
    """This class provides a light model that can be used with biogeochemical models. It takes the incoming short-wave
    radiation as input and computes the photosynthetically active radiation at the water surface and in the water.

    This model can be coupled to another biomodel through GOTMLightModel.couple(model), see below.

    Some biosystem models require temporal light averages, typically over 24 hours.
    These can be computed with this light model with a moving 24 hour time window or as exponential smoothing.

    Args:
        a (Union[float, torch.Tensor]): Fraction of the incoming radiation that is not photosynthetically active.
        g1 (Union[float, torch.Tensor]): Light extinction length for radiation that is not photosynthetically active.
        g2 (Union[float, torch.Tensor]): Light extinction length for photosynthetically active radiation.
        tempmean_method (str, optional): Method for evaluating temporal means of the light fields.
            Possible options: ("running_mean_strict", "exponential_mean"). Defaults to None.
        compute_surface_par (bool, optional): If False, the surface photoactive radiation is set to the value of
            the full light input. If True, it is reduced according to the *a* coefficient. Defaults to True.
        model_name (str, optional): The name of the model. Defaults to "light".
        device (str, optional): The device where the tensors of the model should be stored. Defaults to "cpu".
        dtype (torch.dtype, optional): The default data type of the model. Defaults to torch.float64.
    """

    def __init__(
        self,
        a: Union[float, torch.Tensor],
        g1: Union[float, torch.Tensor],
        g2: Union[float, torch.Tensor],
        tempmean_method: Optional[str] = None,
        compute_surface_par: bool = True,
        model_name: str = "light",
        device="cpu",
        dtype=torch.float64,
        *args,
        **kwargs,
    ):
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        super().__init__(model_name=model_name, device=device, dtype=dtype, *args, **kwargs)

        self.parameters["light_a"] = a if isinstance(a, torch.Tensor) else torch.tensor(a, device=device, dtype=dtype)
        self.parameters["light_g1"] = (
            g1 if isinstance(g1, torch.Tensor) else torch.tensor(g1, device=device, dtype=dtype)
        )
        self.parameters["light_g2"] = (
            g2 if isinstance(g2, torch.Tensor) else torch.tensor(g2, device=device, dtype=dtype)
        )

        self.light_attenuation_indexes = None
        self.coupled_model = None
        self.tempmean_method = tempmean_method
        if tempmean_method == "running_mean_strict":
            self.temporal_mean_par0 = TemporalMean(
                variable="light_par0", period=86400.0, resolution=3600.0, device=device, dtype=dtype, **kwargs
            )
            self.temporal_mean_par = TemporalMean(
                variable="light_par", period=86400.0, resolution=3600.0, device=device, dtype=dtype, **kwargs
            )
            self.dependencies.append("time")
        elif tempmean_method == "exponential_mean":
            self.light_mean_tau = torch.tensor(90500.0, device=device, dtype=dtype)
        self.compute_surface_par = compute_surface_par

        self.dependencies.extend(["I_0", "zeta", "z", "h"])
        self.output_fields.extend(["light_swr", "light_par", "light_par0"])
        if tempmean_method is not None:
            self.output_fields.extend(["light_par_mean", "light_par0_mean"])

    def register_light_coupling(
        self,
        light_attenuation_indexes: List[int],
        light_attenuation_scale_factor: Union[torch.Tensor, float],
        light_attenuation_background: float,
    ):
        """Registers attenuation indexes, scale factors and background for dynamic light attenuation based on the tracer
        state.

        Args:
            light_attenuation_indexes (list of int): The indexes of the tracers that contribute to light attenuation
            light_attenuation_scale_factor (Union[torch.Tensor, float]): The scale factor for light attenuation
                of individual tracers. Can be scalar or vector, in case different tracers have varying contributions.
            light_attenuation_background (float): The light attenuation background
        """
        self.light_attenuation_indexes = light_attenuation_indexes
        self.light_attenuation_scale_factor = light_attenuation_scale_factor
        self.light_attenuation_background = light_attenuation_background

    def couple(self, model: BaseModel):
        """Couples this light model with another biosystem model.

        Args:
            model (BaseModel): The model to which this model should be coupled.

        Raises:
            ValueError: An error is raised in case the coupled model does not have the required fields
                light_attenuation_indexes, light_attenuation_scale_factor, light_attenuation_background

        Returns:
            BaseModel: This model, which is now a coupled model.
        """
        self.coupled_model = model
        self.tracer_names = model.tracer_names
        try:
            self.register_light_coupling(
                model.light_attenuation_indexes,
                model.light_attenuation_scale_factor,
                model.light_attenuation_background,
            )
        except (KeyError, ValueError, TypeError):
            raise ValueError(
                "Coupling the light model did not work!"
                "Please make sure that the coupled model has the attributes"
                "light_attenuation_indexes (list), light_attenuation_scale_factor (float),"
                "and light_attenuation_background (float)."
            )
        self.do_repair_state = model.do_repair_state
        self.get_vertical_movement = model.get_vertical_movement
        self.repair_state = model.repair_state
        return self

    @classmethod
    def create_model_from_yaml_file(
        cls,
        path_fabm: str,
        path_gotm: str,
        tempmean_method: Optional[str] = None,
        device: str = "cpu",
        dtype: Union[str, torch.dtype] = torch.float64,
        *args,
        **kwargs,
    ):
        """Creates a *GOTM_Light_Model* instance from .yaml configuration files. Designed to be compatible with the
        configuration files of GOTM.

        Args:
            path_fabm (str): Path to the FABM .yaml configuration file
            path_gotm (str): Path to the GOTM .yaml configuration file
            tempmean_method (str, optional): Method for evaluating temporal means of the light fields.
                Possible options: ("running_mean_strict", "exponential_mean"). Defaults to None.
            device (str): Torch device. Defaults to "cpu".
            dtype (Union[str, torch.dtype]): Torch data type. Defaults to torch.float64.

        Returns:
            GOTMLightModel: The generated model
        """
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)

        with open(path_fabm, "r") as file_fabm:
            config_fabm = yaml.safe_load(file_fabm)

        _model_detected = False

        model_name = "light"

        for name_instance, config_instance in config_fabm["instances"].items():
            if config_instance["model"] == "gotm/light":
                if _model_detected:
                    raise ValueError(f"More than one light model detected in file {file_fabm}!")
                model_name = name_instance
                _model_detected = True
                try:
                    a = torch.tensor(config_instance["parameters"]["a"], dtype=dtype, device=device)
                except (KeyError, ValueError, TypeError):
                    raise ValueError(
                        f"instances/{name_instance}/parameters/a missing or invalid in {path_fabm}. "
                        + "Expected type float. The suggested default value is 0.58."
                    )
                try:
                    g1 = torch.tensor(config_instance["parameters"]["g1"], dtype=dtype, device=device)
                except (KeyError, ValueError, TypeError):
                    raise ValueError(
                        f"instances/{name_instance}/parameters/g1 missing or invalid in {path_fabm}. "
                        + "Expected type float. The suggested default value is 0.35."
                    )
                try:
                    g2 = torch.tensor(config_instance["parameters"]["g2"], dtype=dtype, device=device)
                except (KeyError, ValueError, TypeError):
                    raise ValueError(
                        f"instances/{name_instance}/parameters/g2 missing or invalid in {path_fabm}. "
                        + "Expected type float. The suggested default value is 23.0."
                    )
        if not _model_detected:
            # model is not specified in fabm.yaml, so use built-in model from gotm.yaml
            with open(path_gotm, "r") as file_gotm:
                config_gotm = yaml.safe_load(file_gotm)
            if "light_extinction" in config_gotm:
                config_light_extinction = config_gotm["light_extinction"]
                method = config_light_extinction["method"]
                if method == "jerlov-i" or method == 1:
                    a = torch.tensor(0.58, dtype=dtype, device=device)
                    g1 = torch.tensor(0.35, dtype=dtype, device=device)
                    g2 = torch.tensor(23.0, dtype=dtype, device=device)
                elif method == "jerlov-1-50m" or method == 2:
                    a = torch.tensor(0.68, dtype=dtype, device=device)
                    g1 = torch.tensor(1.20, dtype=dtype, device=device)
                    g2 = torch.tensor(28.0, dtype=dtype, device=device)
                elif method == "jerlov-ia" or method == 3:
                    a = torch.tensor(0.62, dtype=dtype, device=device)
                    g1 = torch.tensor(0.60, dtype=dtype, device=device)
                    g2 = torch.tensor(20.0, dtype=dtype, device=device)
                elif method == "jerlov-ib" or method == 4:
                    a = torch.tensor(0.67, dtype=dtype, device=device)
                    g1 = torch.tensor(1.0, dtype=dtype, device=device)
                    g2 = torch.tensor(17.0, dtype=dtype, device=device)
                elif method == "jerlov-ii" or method == 5:
                    a = torch.tensor(0.77, dtype=dtype, device=device)
                    g1 = torch.tensor(1.5, dtype=dtype, device=device)
                    g2 = torch.tensor(14.0, dtype=dtype, device=device)
                elif method == "jerlov-iii" or method == 6:
                    a = torch.tensor(0.78, dtype=dtype, device=device)
                    g1 = torch.tensor(1.4, dtype=dtype, device=device)
                    g2 = torch.tensor(7.9, dtype=dtype, device=device)
                else:
                    if config_light_extinction["A"]["method"] == "constant":
                        a = torch.tensor(config_light_extinction["A"]["constant_value"], dtype=dtype, device=device)
                    else:
                        raise NotImplementedError("reading the light coefficient A from file is not implemented yet")
                    if config_light_extinction["g1"]["method"] == "constant":
                        g1 = torch.tensor(config_light_extinction["g1"]["constant_value"], dtype=dtype, device=device)
                    else:
                        raise NotImplementedError("reading the light coefficient g1 from file is not implemented yet")
                    if config_light_extinction["g2"]["method"] == "constant":
                        g2 = torch.tensor(config_light_extinction["g2"]["constant_value"], dtype=dtype, device=device)
                    else:
                        raise NotImplementedError("reading the light coefficient g2 from file is not implemented yet")

            else:
                raise ValueError(
                    f"No light model detected in file {file_fabm} or {file_gotm}!\
                        Please specify a light model according to the following\
                        template:\ninstances:\n  light:\n    model: gotm/light\n    parameters:\n      a   :  0.58  #\
                        default\n      g1  :  0.1  # default\n      g2  : 24.00  # default"
                )

        model = cls(
            a,
            g1,
            g2,
            tempmean_method=tempmean_method,
            compute_surface_par=_model_detected,
            model_name=model_name,
            device=device,
            dtype=dtype,
            *args,
            **kwargs,
        )

        return model

    def get_parameters(self):
        """Retrieve the parameters of the light model. If another model is coupled, retrieve the parameters of both
        models as one TensorDict.

        Returns:
            (TensorDict): The model parameters
        """
        if self.coupled_model is None:
            return super().get_parameters()
        else:
            return super().get_parameters().copy().update(self.coupled_model.get_parameters())

    def update_parameters(self, parameters):
        """Update the parameters of the light model. If another model is coupled, update the parameter of the respective
        model for each new value.

        Args:
            parameters (TensorDict): The model parameters to be updated
        """
        if self.coupled_model is None:
            super().update_parameters(parameters)
        else:
            super().update_parameters(parameters.select(*self.parameters.keys(), strict=False))
            self.coupled_model.update_parameters(parameters.exclude(*self.parameters.keys()))

            self.register_light_coupling(
                self.coupled_model.light_attenuation_indexes,
                self.coupled_model.light_attenuation_scale_factor,
                self.coupled_model.light_attenuation_background,
            )

    def forward(self, concentration: TensorDict, dt: float, dynamic_inputs: TensorDict, *args, **kwargs) -> TensorDict:
        """Compute the NPZD model time step based on the momentary state. Might require additional forcing inputs, e.g.
        incoming radiation.

        Args:
            concentration (TensorDict): The current tracer concentrations. Should have the field:
              - "pelagic": The pelagic tracer concentrations. [batch, tracer, nlev]
            dt (float): The time step of the simulation.
            dynamic_inputs (TensorDict): Dict of inputs. Should have the fields:
              - "I_0": Incoming surface radiation. [batch]
              - "zeta": Water surface elevation. [batch]
              - "z": Vertical coordinates at cell centers. [batch, nlev]
              - "h": Vertical layer heights. [batch, 1, nlev] or [batch, nlev]

        Returns:
            TensorDict: The new tracer concentration field. Adds/updates the fields:
              - "light_swr": Shortwave radiation. [batch, nlev]
              - "light_par": Photosynthetically active radiation. [batch, nlev]
              - "light_par0": Surface photosynthetically active radiation. [batch, nlev]
        """
        concentration = self.compute_light(concentration, dt, dynamic_inputs, *args, **kwargs)

        if self.coupled_model is not None:
            concentration = self.coupled_model(concentration, dt, dynamic_inputs, *args, **kwargs)

        return concentration

    def compute_light(
        self, concentration: TensorDict, dt: float, dynamic_inputs: TensorDict, *args, **kwargs
    ) -> TensorDict:
        """Compute the light distribution based on incoming surface radiation and water and tracer attenuation.

        Args:
            concentration (TensorDict): The current tracer concentrations. Should have the field:
              - "pelagic": The pelagic tracer concentrations. [batch, tracer, nlev]
            dt (float): The time step of the simulation.
            dynamic_inputs (TensorDict): Dict of inputs. Should have the fields:
              - "I_0": Incoming surface radiation. [batch]
              - "zeta": Water surface elevation. [batch]
              - "z": Vertical coordinates at cell centers. [batch, nlev]
              - "h": Vertical layer heights. [batch, 1, nlev] or [batch, nlev]

        Returns:
            TensorDict: The new tracer concentration field. Adds/updates the fields:
              - "light_swr": Shortwave radiation. [batch, nlev]
              - "light_par": Photosynthetically active radiation. [batch, nlev]
              - "light_par0": Surface photosynthetically active radiation. [batch, nlev]
        """
        swr0 = dynamic_inputs["I_0"]

        zeta = dynamic_inputs["zeta"]
        z = dynamic_inputs["z"]
        h = dynamic_inputs["h"]
        if h.ndim > z.ndim:
            h = h[..., 0, :]
        if zeta.ndim < z.ndim:
            zeta = zeta.unsqueeze(-1)
        z = z - zeta

        if self.light_attenuation_indexes is not None:
            ext = concentration["pelagic"][..., self.light_attenuation_indexes, :]
            # [..., tracer, z]
            ext = ext * self.light_attenuation_scale_factor
            ext = torch.sum(ext, dim=-2) + self.light_attenuation_background
            # [..., z]
            # z index starts at bottom and goes up
            ext = ext * h

            ext_cumsum = torch.cumsum(ext.flip(dims=(-1,)), dim=-1).flip(dims=(-1,))

            bioext = ext_cumsum - 0.5 * ext
        else:
            bioext = 0.0

        if self.compute_surface_par:
            par0 = swr0 * (1 - self.parameters["light_a"])
            par = par0.unsqueeze(-1) * torch.exp(z / self.parameters["light_g2"] - bioext)
        else:
            par0 = swr0
            par = (
                par0.unsqueeze(-1)
                * (1 - self.parameters["light_a"])
                * torch.exp(z / self.parameters["light_g2"] - bioext)
            )
        swr = par + swr0.unsqueeze(-1) * self.parameters["light_a"] * torch.exp(z / self.parameters["light_g1"])

        concentration["light_swr"] = swr
        concentration["light_par"] = par
        concentration["light_par0"] = par0

        if self.tempmean_method == "running_mean_strict":
            concentration = self.temporal_mean_par0.forward(concentration, dt, dynamic_inputs)
            concentration = self.temporal_mean_par.forward(concentration, dt, dynamic_inputs)
        elif self.tempmean_method == "exponential_mean":
            one_minus_alpha = torch.exp(-dt / self.light_mean_tau)
            alpha = 1.0 - one_minus_alpha
            if "light_par_mean" in concentration:
                concentration["light_par_mean"] = one_minus_alpha * concentration["light_par_mean"] + alpha * par
            else:
                concentration["light_par_mean"] = par
            if "light_par0_mean" in concentration:
                concentration["light_par0_mean"] = one_minus_alpha * concentration["light_par0_mean"] + alpha * par0
            else:
                concentration["light_par0_mean"] = par0

        return concentration

    __call__ = forward
