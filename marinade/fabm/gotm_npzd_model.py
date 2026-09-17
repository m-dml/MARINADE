from typing import Union

import torch
import yaml
from marinade.fabm.bio_model import BioModel
from tensordict import TensorDict


class GOTMNPZDModel(BioModel):
    """This class describes a toy biosystem model with four tracers: Nutrients, phytoplankton, zooplankton and
    detritus.

    If created by the constructor method, this model uses the same default parameters as the version of this model
    provided by FABM. A model with custom parameters can be created from a .yaml file with
    *GOTMNPZDModel.create_model_from_yaml_file*.

    Args:
        ode_method (str): The numerical method for solving the tracer ODE. Supported schemes:
            ("FE": Forward Euler, "RK2": 2nd order Runge-Kutta, "RK4": 4th order Runge-Kutta)
        model_name (str, optional): The name of the model. Defaults to "npzd".
        device (str, optional): The device where the tensors of the model should be stored. Defaults to "cpu".
        dtype (torch.dtype, optional): The default data type of the model. Defaults to torch.float64.
    """

    def __init__(
        self,
        ode_method: str,
        model_name: str = "npzd",
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        *args,
        **kwargs,
    ):
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)

        super().__init__(ode_method=ode_method, model_name=model_name, device=device, dtype=dtype, *args, **kwargs)

        self.tracer_names.extend(
            [f"{self.model_name}_nut", f"{self.model_name}_phy", f"{self.model_name}_zoo", f"{self.model_name}_det"]
        )

        self.parameters["p0"] = torch.tensor(
            0.0225, device=device, dtype=dtype
        )  # background phytoplankton concentration (mmol m-3), default = 0.0225
        self.parameters["z0"] = torch.tensor(
            0.0225, device=device, dtype=dtype
        )  # background zooplankton concentration (mmol m-3), default = 0.0225
        self.parameters["kc"] = torch.tensor(
            0.03, device=device, dtype=dtype
        )  # specific light extinction of phytoplankton and detritus (m2 mmol-1), default = 0.03
        self.parameters["i_min"] = torch.tensor(
            25.0, device=device, dtype=dtype
        )  # minimum light intensity in euphotic zone (W m-2), default = 25.0
        self.parameters["rmax"] = torch.tensor(
            1.0 / 86400, device=device, dtype=dtype
        )  # maximum specific growth rate of phytoplankton (d-1), default = 1.0
        self.parameters["gmax"] = torch.tensor(
            0.2 / 86400, device=device, dtype=dtype
        )  # maximum specific grazing rate of zooplankton (d-1), default = 0.5
        self.parameters["iv"] = torch.tensor(
            1.1, device=device, dtype=dtype
        )  # Ivlev grazing constant (m3 mmol-1), default = 1.1
        self.parameters["alpha"] = torch.tensor(
            1.35, device=device, dtype=dtype
        )  # half-saturation nutrient concentration for phytoplankton (mmol m-3), default = 0.3
        self.parameters["rpn"] = torch.tensor(
            0.01 / 86400, device=device, dtype=dtype
        )  # loss rate of phytoplankton to nutrients (d-1), default = 0.01
        self.parameters["rzn"] = torch.tensor(
            0.01 / 86400, device=device, dtype=dtype
        )  # loss rate of zooplankton to nutrients (d-1), default = 0.01
        self.parameters["rdn"] = torch.tensor(
            0.003 / 86400, device=device, dtype=dtype
        )  # detritus remineralization rate (d-1), default = 0.003
        self.parameters["rpdu"] = torch.tensor(
            0.02 / 86400, device=device, dtype=dtype
        )  # phytoplankton mortality in euphotic zone (d-1), default = 0.02
        self.parameters["rpdl"] = torch.tensor(
            0.1 / 86400, device=device, dtype=dtype
        )  # phytoplankton mortality below euphotic zone (d-1), default = 0.1
        self.parameters["rzd"] = torch.tensor(
            0.02 / 86400, device=device, dtype=dtype
        )  # zooplankton mortality (d-1), default = 0.02
        self.parameters["dic_per_n"] = torch.tensor(
            106.0 / 16.0, device=device, dtype=dtype
        )  # molar C:N ratio of biomass, default=106.0_rk/16.0_rk)
        self.parameters["w_p"] = torch.tensor(
            -1.0 / 86400, device=device, dtype=dtype
        )  # vertical velocity of phytoplankton (<0 for sinking) (m d-1), default = -1.0
        self.parameters["w_d"] = torch.tensor(
            -5.0 / 86400, device=device, dtype=dtype
        )  # vertical velocity of detritus (<0 for sinking) (m d-1), default = -5.0

        self.ws = torch.tensor(
            [[0.0, self.parameters["w_p"], 0.0, self.parameters["w_d"]]], device=device, dtype=dtype
        ).unsqueeze(-1)

        self.minimum = torch.tensor([0.0, 0.0, 0.0, 0.0], device=device, dtype=dtype).unsqueeze(0).unsqueeze(-1)
        self.minimum_td = TensorDict({"pelagic": self.minimum})

        self.light_attenuation_indexes = [1, 3]
        self.light_attenuation_scale_factor = self.parameters["kc"]
        self.light_attenuation_background = self.parameters["p0"] * self.parameters["kc"]

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
        """Creates a GOTMNPZDModel instance from .yaml configuration files. Should generally be used instead of the
        constructor. Designed to be compatible with the configuration files of GOTM.

        Should be overwritten by subclasses.

        Args:
            path_fabm (str): Path to the FABM .yaml configuration file
            path_gotm (str): Path to the GOTM .yaml configuration file
            device (str, optional): Torch device. Defaults to "cpu".
            dtype (Union[str, torch.dtype], optional): Torch data type. Defaults to torch.float64.

        Returns:
            GOTMNPZDModel: The generated model
        """
        _parameters = ["p0", "z0", "kc", "i_min", "iv", "alpha"]  # , 'dic_per_n']
        _parameters_daily = ["rmax", "gmax", "rpn", "rzn", "rdn", "rpdu", "rpdl", "rzd", "w_p", "w_d"]
        _seconds_per_day = 86400.0

        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)

        # load the GOTM and FABM .yaml config files:
        with open(path_gotm, "r") as file_gotm:
            config_gotm = yaml.safe_load(file_gotm)
        with open(path_fabm, "r") as file_fabm:
            config_fabm = yaml.safe_load(file_fabm)

        try:
            ode_method = config_gotm["fabm"]["numerics"]["ode_method"]
        except (KeyError, ValueError, TypeError):
            raise ValueError(f"fabm/numerics/ode_method missing or invalid in {path_gotm}.")

        _model_detected = False

        for name_instance, config_instance in config_fabm["instances"].items():
            if config_instance["model"] == "gotm/npzd":
                if _model_detected:
                    raise ValueError(f"More than one NPZD model detected in file {file_fabm}!")
                _model_detected = True

                model = cls(
                    device=device, ode_method=ode_method, model_name=name_instance, dtype=dtype, *args, **kwargs
                )
                for _name in _parameters:
                    try:
                        model.parameters[_name] = torch.tensor(
                            config_instance["parameters"][_name], device=device, dtype=dtype
                        )
                    except (KeyError, ValueError, TypeError):
                        raise ValueError(
                            f"instances/{name_instance}/parameters/{_name} missing or invalid\
                                        in {path_fabm}. Expected type float."
                        )
                for _name in _parameters_daily:
                    try:
                        model.parameters[_name] = torch.tensor(
                            config_instance["parameters"][_name] / _seconds_per_day, device=device, dtype=dtype
                        )
                    except (KeyError, ValueError, TypeError):
                        raise ValueError(
                            f"instances/{name_instance}/parameters/{_name} missing or invalid\
                                        in {path_fabm}. Expected type float."
                        )
        if not _model_detected:
            raise ValueError(
                f"No NPZD model detected in file {file_fabm}! Please specify an NPZD model according to the following\
                    template:\ninstances:\n  npzd:\n    model: gotm/npzd\n    parameters:\n      p0  :  0.0225  #\
                    default\n      z0  :  0.0225  # default\n      ..."
            )

        model.ws = (
            torch.stack(
                [
                    torch.zeros_like(model.parameters["w_p"]),
                    model.parameters["w_p"],
                    torch.zeros_like(model.parameters["w_d"]),
                    model.parameters["w_d"],
                ]
            )
            .unsqueeze(0)
            .unsqueeze(-1)
        )
        model.light_attenuation_scale_factor = model.parameters["kc"]
        model.light_attenuation_background = model.parameters["p0"] * model.parameters["kc"]

        return model

    def update_parameters(self, parameters: TensorDict):
        """Update the parameters of the model.

        Args:
            parameters (TensorDict): The model parameters to be updated
        """
        super().update_parameters(parameters)

        _zero = torch.zeros_like(self.parameters["w_p"])
        if _zero.ndim == 0:
            self.ws = torch.stack([_zero, self.parameters["w_p"], _zero, self.parameters["w_d"]])[None, :, None]
        elif _zero.ndim > 2:
            self.ws = torch.cat([_zero, self.parameters["w_p"], _zero, self.parameters["w_d"]], dim=-2)
        else:
            raise Exception(
                "Invalid shape for sinking rate parameter w_p! It should be scalar or of shape [batch, 1, z]."
            )

        self.light_attenuation_scale_factor = self.parameters["kc"]
        self.light_attenuation_background = self.parameters["p0"] * self.parameters["kc"]

    @staticmethod
    def fnp(rmax, alpha, n, p, par, iopt):
        """Helper method for phytoplankton growth, see GOTM/FABM source code and documentation."""
        fnp = rmax * par / iopt * torch.exp(1.0 - par / iopt) * n / (alpha + n) * p
        return fnp

    @staticmethod
    def fpz(gmax, iv, p, z):
        """Helper method for zooplankton grazing, see GOTM/FABM source code and documentation."""
        fpz = gmax * (1.0 - torch.exp(-iv * iv * p * p)) * z
        return fpz

    def do(
        self, concentration: Union[TensorDict, torch.Tensor], dynamic_inputs: TensorDict, *args, **kwargs
    ) -> Union[TensorDict, torch.Tensor]:
        """Calculate right-hand side of biological time step ODE as a vector with temporal derivatives.

        Returns:
            Union[TensorDict, torch.Tensor]: Vector of temporal derivatives
        """

        use_state_light = False

        if isinstance(concentration, torch.Tensor):
            x = concentration
        elif isinstance(concentration, TensorDict):
            x = concentration["pelagic"]
            use_state_light = "light_par" in concentration

        n = x[..., 0, :]
        p = x[..., 1, :]
        z = x[..., 2, :]
        d = x[..., 3, :]

        if use_state_light:
            par0 = concentration["light_par0"]
            par = concentration["light_par"]
        else:
            par0 = dynamic_inputs["par0"]
            par = dynamic_inputs["par"]

        if par0.ndim < par.ndim:
            par0 = par0.unsqueeze(-1)

        iopt = torch.clamp_min(0.25 * par0, self.parameters["i_min"])

        rpd = self.mask_func(self.parameters["i_min"] - par, self.parameters["rpdl"], self.parameters["rpdu"])

        primprod = self.fnp(self.parameters["rmax"], self.parameters["alpha"], n, p + self.parameters["p0"], par, iopt)
        g = self.fpz(self.parameters["gmax"], self.parameters["iv"], p, z + self.parameters["z0"])

        dn = -primprod + self.parameters["rpn"] * p + self.parameters["rzn"] * z + self.parameters["rdn"] * d
        dp = primprod - g - self.parameters["rpn"] * p - rpd * p
        dz = g - self.parameters["rzn"] * z - self.parameters["rzd"] * z
        dd = rpd * p + self.parameters["rzd"] * z - self.parameters["rdn"] * d

        dx = torch.stack([dn, dp, dz, dd], axis=-2)

        if isinstance(concentration, TensorDict):
            return TensorDict({"pelagic": dx}, batch_size=concentration.batch_size)
        else:
            return dx

    def do_ppdd(self):
        """Calculate source/sink terms of biological variables as production and destruction matrices.

        Raises:
            NotImplementedError: This method still has to be implemented.
        """
        raise NotImplementedError()
