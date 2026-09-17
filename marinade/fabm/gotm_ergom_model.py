from typing import Union

import torch
import yaml
from marinade.fabm.bio_model import BioModel
from tensordict import TensorDict


class GOTMErgomModel(BioModel):
    """This class describes the biosystem model ERGOM introduced by Neumann and Fennel.

    If created by the constructor method, this model uses the same default parameters as the version of this model
    provided by FABM. A model with custom parameters can be created from a .yaml file with
    *GOTMErgomModel.create_model_from_yaml_file*.

    Args:
        ode_method (str): The numerical method for solving the tracer ODE. Supported schemes:
            ("FE": Forward Euler, "RK2": 2nd order Runge-Kutta, "RK4": 4th order Runge-Kutta)
        fluff (bool, optional): Whether to use an additional benthic tracer. Defaults to False.
        model_name (str, optional): The name of the model. Defaults to "ergom".
        device (str, optional): The device where the tensors of the model should be stored. Defaults to "cpu".
        dtype (torch.dtype, optional): The default data type of the model. Defaults to torch.float64.
    """

    def __init__(
        self,
        ode_method: str,
        fluff: bool = False,
        model_name: str = "ergom",
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        *args,
        **kwargs,
    ):
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)

        super().__init__(ode_method=ode_method, model_name=model_name, device=device, dtype=dtype, *args, **kwargs)

        self.parameters["sfl_po"] = torch.tensor(0.0015 / 86400, device=device, dtype=dtype)
        self.parameters["sfl_ni"] = torch.tensor(0.09 / 86400, device=device, dtype=dtype)
        self.parameters["sfl_am"] = torch.tensor(0.07 / 86400, device=device, dtype=dtype)
        self.fluff = fluff
        self.parameters["p10"] = torch.tensor(0.0225, device=device, dtype=dtype)
        self.parameters["p20"] = torch.tensor(0.0225, device=device, dtype=dtype)
        self.parameters["p30"] = torch.tensor(0.0225, device=device, dtype=dtype)
        self.parameters["zo0"] = torch.tensor(0.0225, device=device, dtype=dtype)
        self.parameters["kc"] = torch.tensor(0.03, device=device, dtype=dtype)
        self.parameters["i_min"] = torch.tensor(25.0, device=device, dtype=dtype)
        self.parameters["r1max"] = torch.tensor(2.0 / 86400, device=device, dtype=dtype)
        self.parameters["r2max"] = torch.tensor(0.7 / 86400, device=device, dtype=dtype)
        self.parameters["r3max"] = torch.tensor(0.5 / 86400, device=device, dtype=dtype)
        self.parameters["alpha1"] = torch.tensor(1.35, device=device, dtype=dtype)
        self.parameters["alpha2"] = torch.tensor(0.675, device=device, dtype=dtype)
        self.parameters["alpha3"] = torch.tensor(0.5, device=device, dtype=dtype)
        self.parameters["lpa"] = torch.tensor(0.01 / 86400, device=device, dtype=dtype)
        self.parameters["lpd"] = torch.tensor(0.02 / 86400, device=device, dtype=dtype)
        self.parameters["tf"] = torch.tensor(10.0, device=device, dtype=dtype)
        self.parameters["tbg"] = torch.tensor(14.0, device=device, dtype=dtype)
        self.parameters["beta_bg"] = torch.tensor(1.0, device=device, dtype=dtype)
        self.parameters["g1max"] = torch.tensor(0.5 / 86400, device=device, dtype=dtype)
        self.parameters["g2max"] = torch.tensor(0.5 / 86400, device=device, dtype=dtype)
        self.parameters["g3max"] = torch.tensor(0.25 / 86400, device=device, dtype=dtype)
        self.parameters["lza"] = torch.tensor(0.0666666666 / 86400, device=device, dtype=dtype)
        self.parameters["lzd"] = torch.tensor(0.1333333333 / 86400, device=device, dtype=dtype)
        self.parameters["iv"] = torch.tensor(0.24444444, device=device, dtype=dtype)
        self.parameters["topt"] = torch.tensor(20.0, device=device, dtype=dtype)
        self.parameters["lan"] = torch.tensor(0.1 / 86400, device=device, dtype=dtype)
        self.parameters["oan"] = torch.tensor(0.01, device=device, dtype=dtype)
        self.parameters["beta_an"] = torch.tensor(0.11, device=device, dtype=dtype)
        self.parameters["lda"] = torch.tensor(0.003 / 86400, device=device, dtype=dtype)
        self.parameters["tda"] = torch.tensor(13.0, device=device, dtype=dtype)
        self.parameters["beta_da"] = torch.tensor(20.0, device=device, dtype=dtype)
        self.parameters["pvel"] = torch.tensor(5.0 / 86400, device=device, dtype=dtype)
        self.parameters["sr"] = torch.tensor(0.0625, device=device, dtype=dtype)
        self.parameters["s1"] = torch.tensor(5.3, device=device, dtype=dtype)
        self.parameters["s2"] = torch.tensor(6.625, device=device, dtype=dtype)
        self.parameters["s3"] = torch.tensor(8.625, device=device, dtype=dtype)
        self.parameters["s4"] = torch.tensor(2.0, device=device, dtype=dtype)
        self.parameters["a0"] = torch.tensor(31.25, device=device, dtype=dtype)
        self.parameters["a1"] = torch.tensor(14.603, device=device, dtype=dtype)
        self.parameters["a2"] = torch.tensor(0.4025, device=device, dtype=dtype)
        if self.fluff:
            self.parameters["lds"] = torch.tensor(3.5 / 86400, device=device, dtype=dtype)
            self.parameters["lsd"] = torch.tensor(25.0 / 86400, device=device, dtype=dtype)
            self.parameters["tau_crit"] = torch.tensor(0.07, device=device, dtype=dtype)
            self.parameters["lsa"] = torch.tensor(0.001 / 86400, device=device, dtype=dtype)
            self.parameters["bsa"] = torch.tensor(0.15, device=device, dtype=dtype)
            self.parameters["ph1"] = torch.tensor(0.15, device=device, dtype=dtype)
            self.parameters["ph2"] = torch.tensor(0.1, device=device, dtype=dtype)
        self.parameters["w_p1"] = torch.tensor(-1.0 / 86400, device=device, dtype=dtype)
        self.parameters["w_p2"] = torch.tensor(-5.0 / 86400, device=device, dtype=dtype)
        self.parameters["w_p3"] = torch.tensor(-5.0 / 86400, device=device, dtype=dtype)
        self.parameters["w_de"] = torch.tensor(-3.0 / 86400, device=device, dtype=dtype)

        self.tracer_names.extend(
            [
                f"{self.model_name}_dia",
                f"{self.model_name}_fla",
                f"{self.model_name}_cya",
                f"{self.model_name}_zoo",
                f"{self.model_name}_det",
                f"{self.model_name}_amm",
                f"{self.model_name}_nit",
                f"{self.model_name}_pho",
                f"{self.model_name}_oxy",
            ]
        )
        if self.fluff:
            self.benthic_tracer_names.append(f"{self.model_name}_flf")

        self.dependencies.extend(["temp", "salt", "taub", "wind"])

        self.parameters["wo"] = torch.tensor(30.0, device=device, dtype=dtype)
        self.parameters["wn"] = torch.tensor(0.1, device=device, dtype=dtype)

        # constants for osat_weiss:
        self.aa1 = torch.tensor(-173.4292, device=device, dtype=dtype)
        self.aa2 = torch.tensor(249.6339, device=device, dtype=dtype)
        self.a3 = torch.tensor(143.3483, device=device, dtype=dtype)
        self.a4 = torch.tensor(-21.8492, device=device, dtype=dtype)
        self.b1 = torch.tensor(-0.033096, device=device, dtype=dtype)
        self.b2 = torch.tensor(0.014259, device=device, dtype=dtype)
        self.b3 = torch.tensor(-0.001700, device=device, dtype=dtype)
        self.kelvin = torch.tensor(273.16, device=device, dtype=dtype)
        self.mol_per_liter = torch.tensor(44.661, device=device, dtype=dtype)

        # constants for do_bottom
        self.dot2 = torch.tensor(0.2, device=device, dtype=dtype)

        self.setup_internal_fields(device=device, dtype=dtype)

    def setup_internal_fields(self, device: str, dtype: torch.dtype):
        """Routine for setting up internal fields that depend on internal parameters."""

        _zero = torch.zeros_like(self.parameters["w_p1"], requires_grad=False)
        _ws = [
            self.parameters["w_p1"],
            self.parameters["w_p2"],
            self.parameters["w_p3"],
            _zero,
            self.parameters["w_de"],
            _zero,
            _zero,
            _zero,
            _zero,
        ]
        self.ws = torch.stack(_ws).unsqueeze(0).unsqueeze(-1)

        _minimum = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -torch.inf]
        self.minimum = torch.tensor(_minimum, device=device, dtype=dtype).unsqueeze(0).unsqueeze(-1)

        self.minimum_td = TensorDict({"pelagic": self.minimum, "benthic": torch.zeros((1), dtype=dtype, device=device)})

        self.light_attenuation_indexes = [0, 1, 2, 4]
        self.light_attenuation_scale_factor = self.parameters["kc"]
        self.light_attenuation_background = (
            self.parameters["p10"] + self.parameters["p20"] + self.parameters["p30"]
        ) * self.parameters["kc"]

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
        """Creates a GOTMErgomModel instance from .yaml configuration files. Should generally be used instead of the
        constructor. Designed to be compatible with the configuration files of GOTM.

        Args:
            path_fabm (str): Path to the FABM .yaml configuration file
            path_gotm (str): Path to the GOTM .yaml configuration file
            device (str, optional): Torch device. Defaults to "cpu".
            dtype (str or torch.dtype, optional): Torch data type. Defaults to torch.float64.

        Returns:
            GOTMErgomModel: The generated model
        """
        _parameters = [
            "p10",
            "p20",
            "p30",
            "zo0",
            "kc",
            "i_min",
            "alpha1",
            "alpha2",
            "alpha3",
            "tf",
            "tbg",
            "beta_bg",
            "iv",
            "topt",
            "oan",
            "beta_an",
            "tda",
            "beta_da",
            "sr",
            "s1",
            "s2",
            "s3",
            "s4",
            "a0",
            "a1",
            "a2",
        ]  # , 'dic_per_n']
        _parameters_daily = [
            "sfl_po",
            "sfl_ni",
            "sfl_am",
            "r1max",
            "r2max",
            "r3max",
            "lpa",
            "lpd",
            "g1max",
            "g2max",
            "g3max",
            "lza",
            "lzd",
            "lan",
            "lda",
            "pvel",
            "w_p1",
            "w_p2",
            "w_p3",
            "w_de",
        ]
        _parameters_fluff = ["tau_crit", "bsa", "ph1", "ph2"]
        _parameters_fluff_daily = ["lds", "lsd", "lsa"]
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

        try:
            repair_state = config_gotm["fabm"]["repair_state"]
        except (KeyError, ValueError, TypeError):
            raise ValueError(f"fabm/repair_state missing or invalid in {path_gotm}.")

        model = None  # cls(device=device, ode_method=ode_method, dtype=dtype, *args, **kwargs)

        for name_instance, config_instance in config_fabm["instances"].items():
            if config_instance["model"] == "gotm/ergom":
                if model is not None:
                    raise ValueError(f"More than one NPZD model detected in file {file_fabm}!")

                try:
                    fluff = config_instance["parameters"]["fluff"]
                except (KeyError, ValueError, TypeError):
                    raise ValueError(
                        f"instances/{name_instance}/parameters/fluff missing or invalid "
                        f"in {path_fabm}. Expected type bool."
                    )
                model = cls(
                    fluff=fluff,
                    model_name=name_instance,
                    device=device,
                    ode_method=ode_method,
                    repair_state=repair_state,
                    dtype=dtype,
                    *args,
                    **kwargs,
                )
                if fluff:
                    _parameters.extend(_parameters_fluff)
                    _parameters_daily.extend(_parameters_fluff_daily)

                for _name in _parameters:
                    try:
                        model.parameters[_name] = torch.tensor(
                            config_instance["parameters"][_name], device=device, dtype=dtype
                        )
                    except (KeyError, ValueError, TypeError):
                        print("Printing missing parameters for yaml file:")
                        for name0 in _parameters:
                            print(f"      {name0}: {getattr(model, name0)}  # default = {getattr(model, name0)}")
                        for name0 in _parameters_daily:
                            print(
                                f"      {name0}: {getattr(model, name0) * _seconds_per_day}  "
                                + f"# default = {getattr(model, name0) * _seconds_per_day}"
                            )
                        raise ValueError(
                            f"instances/{name_instance}/parameters/{_name} missing or invalid "
                            f"in {path_fabm}. Expected type float."
                        )
                for _name in _parameters_daily:
                    try:
                        model.parameters[_name] = torch.tensor(
                            config_instance["parameters"][_name] / _seconds_per_day, device=device, dtype=dtype
                        )
                    except (KeyError, ValueError, TypeError):
                        print("Printing missing parameters for yaml file:")
                        for name0 in _parameters:
                            print(f"      {name0}: {getattr(model, name0)}  # default = {getattr(model, name0)}")
                        for name0 in _parameters_daily:
                            print(
                                f"      {name0}: {getattr(model, name0) * _seconds_per_day}  "
                                + f"# default = {getattr(model, name0) * _seconds_per_day}"
                            )
                        raise ValueError(
                            f"instances/{name_instance}/parameters/{_name} missing or invalid "
                            f"in {path_fabm}. Expected type float."
                        )
        if model is None:
            raise ValueError(
                f"No NPZD model detected in file {file_fabm}! Please specify an NPZD model according to the following\
                    template:\ninstances:\n  npzd:\n    model: gotm/npzd\n    parameters:\n      p0  :  0.0225  #\
                    default\n      z0  :  0.0225  # default\n      ..."
            )

        model.setup_internal_fields(device=device, dtype=dtype)

        return model

    def update_parameters(self, parameters: TensorDict):
        """Update the parameters of the model.

        Args:
            parameters (TensorDict): The model parameters to be updated
        """
        super().update_parameters(parameters)

        if any(k in parameters for k in ("w_p1", "w_p2", "w_p3", "w_de")):
            _zero = torch.zeros_like(self.parameters["w_p1"], requires_grad=False)
            _ws = [
                self.parameters["w_p1"],
                self.parameters["w_p2"],
                self.parameters["w_p3"],
                _zero,
                self.parameters["w_de"],
                _zero,
                _zero,
                _zero,
                _zero,
            ]
            self.ws = torch.stack(_ws).unsqueeze(0).unsqueeze(-1)

        if "kc" in parameters:
            self.light_attenuation_scale_factor = self.parameters["kc"]
        if any(k in parameters for k in ("p10", "p20", "p30", "kc")):
            self.light_attenuation_background = (
                self.parameters["p10"] + self.parameters["p20"] + self.parameters["p30"]
            ) * self.parameters["kc"]

    def th(self, x, w, min, max):
        r"""Helper method: Smooth approximation of heaviside function."""
        if isinstance(w, torch.Tensor):
            return self.mask_func(
                w - 1.0e-10,
                min
                + (max - min)
                * 0.5
                * (1.0 + torch.tanh(x / torch.clamp_min(w, 1.0e-10))),  # prevent nan values in gradient
                self.mask_func(x, 1.0, 0.0),
            )
        else:
            if w > 1.0e-10:
                return min + (max - min) * 0.5 * (1.0 + torch.tanh(x / w))
            else:
                return self.mask_func(x, 1.0, 0.0)

    @staticmethod
    def yy(a, x):
        """Helper method: Squared Michaelis-Menten limiter for nutrient uptake.

        See GOTM/FABM source code and documentation for further information.
        """
        return x * x / (a * a + x * x)

    @staticmethod
    def fpz(iv, g, t, topt, psum):
        """Helper method for calculating zooplankton grazing, see GOTM/FABM source code and documentation."""
        t_topt = t / topt
        return g * (1.0 + t_topt * t_topt * torch.exp(1.0 - 2.0 * t_topt)) * (1.0 - torch.exp(-iv * iv * psum * psum))

    def osat_weiss(self, t, s):
        """Helper method for calculating oxygen saturation."""
        tk = (t + self.kelvin) * 0.01
        return (
            torch.exp(
                self.aa1
                + self.aa2 / tk
                + self.a3 * torch.log(tk)
                + self.a4 * tk
                + s * (self.b1 + (self.b2 + self.b3 * tk) * tk)
            )
            * self.mol_per_liter
        )

    def do(
        self, concentration: Union[TensorDict, torch.Tensor], dynamic_inputs: TensorDict, *args, **kwargs
    ) -> Union[TensorDict, torch.Tensor]:
        """Calculate right-hand side of biological time step ODE as a vector with temporal derivatives.

        Args:
            concentration (Union[TensorDict, torch.Tensor]): The current system state.
                Should have the fields "pelagic" and "benthic".
                If this model is coupled with a light model, *concentration* should also
                have the fields "light_par" and "light_par0".
            dynamic_inputs (TensorDict): Time-dependent model inputs. Should have the fields "taub", "temp", "salt".
                If this model is not coupled to a light model, should also have the corresponding light fields.

        Returns:
            Union[TensorDict, torch.Tensor]: Vector of temporal derivatives
        """

        use_state_light = False

        if isinstance(concentration, torch.Tensor):
            x = concentration
        elif isinstance(concentration, TensorDict):
            x = concentration["pelagic"]
            use_state_light = "light_par" in concentration

        p1 = x[..., 0, :]
        p2 = x[..., 1, :]
        p3 = x[..., 2, :]
        zo = x[..., 3, :]
        de = x[..., 4, :]
        am = x[..., 5, :]
        ni = x[..., 6, :]
        po = x[..., 7, :]
        o2 = x[..., 8, :]

        if use_state_light:
            par0 = concentration["light_par0"]
            par = concentration["light_par"]
        else:
            par0 = dynamic_inputs["par0"]
            par = dynamic_inputs["par"]

        if par0.ndim < par.ndim:
            par0 = par0.unsqueeze(-1)

        temp = dynamic_inputs["temp"]

        iopt = torch.clamp_min(0.25 * par0, self.parameters["i_min"])
        par_iopt = par / iopt
        ppi = par_iopt * torch.exp(1.0 - par_iopt)

        thopnp = self.th(o2, self.parameters["wo"], 0.0, 1.0) * self.yy(self.parameters["wn"], ni)
        thomnp = self.th(-o2, self.parameters["wo"], 0.0, 1.0) * self.yy(self.parameters["wn"], ni)
        thomnm = self.th(-o2, self.parameters["wo"], 0.0, 1.0) * (1.0 - self.yy(self.parameters["wn"], ni))
        thsum = thopnp + thomnp + thomnm
        thopnp = thopnp / thsum
        thomnp = thomnp / thsum
        thomnm = thomnm / thsum

        psum = p1 + p2 + p3 + self.parameters["p10"] + self.parameters["p20"] + self.parameters["p30"]

        llda = self.parameters["lda"] * (1.0 + self.parameters["beta_da"] * self.yy(self.parameters["tda"], temp))
        llan = (
            self.th(o2, 0.0, 0.0, 1.0)
            * o2
            / (self.parameters["oan"] + o2)
            * self.parameters["lan"]
            * torch.exp(self.parameters["beta_an"] * temp)
        )

        lp = self.parameters["lpa"] + self.parameters["lpd"]

        r1 = (
            self.parameters["r1max"]
            * torch.minimum(
                self.yy(self.parameters["alpha1"], am + ni),
                torch.minimum(self.yy(self.parameters["sr"] * self.parameters["alpha1"], po), ppi),
            )
            * (p1 + self.parameters["p10"])
        )
        r2 = (
            self.parameters["r2max"]
            * (1.0 + self.yy(self.parameters["tf"], temp))
            * torch.minimum(
                self.yy(self.parameters["alpha2"], am + ni),
                torch.minimum(self.yy(self.parameters["sr"] * self.parameters["alpha2"], po), ppi),
            )
            * (p2 + self.parameters["p20"])
        )
        r3 = (
            self.parameters["r3max"]
            * 1.0
            / (1.0 + torch.exp(self.parameters["beta_bg"] * (self.parameters["tbg"] - temp)))
            * torch.minimum(self.yy(self.parameters["sr"] * self.parameters["alpha3"], po), ppi)
            * (p3 + self.parameters["p30"])
        )

        dp1 = (
            r1
            - self.fpz(self.parameters["iv"], self.parameters["g1max"], temp, self.parameters["topt"], psum)
            * p1
            / psum
            * (zo + self.parameters["zo0"])
            - lp * p1
        )
        dp2 = (
            r2
            - self.fpz(self.parameters["iv"], self.parameters["g2max"], temp, self.parameters["topt"], psum)
            * p2
            / psum
            * (zo + self.parameters["zo0"])
            - lp * p2
        )
        dp3 = (
            r3
            - self.fpz(self.parameters["iv"], self.parameters["g3max"], temp, self.parameters["topt"], psum)
            * p3
            / psum
            * (zo + self.parameters["zo0"])
            - lp * p3
        )
        dzo = (
            (
                self.fpz(self.parameters["iv"], self.parameters["g1max"], temp, self.parameters["topt"], psum) * p1
                + self.fpz(self.parameters["iv"], self.parameters["g2max"], temp, self.parameters["topt"], psum) * p2
                + self.fpz(self.parameters["iv"], self.parameters["g3max"], temp, self.parameters["topt"], psum) * p3
            )
            * (zo + self.parameters["zo0"])
            / psum
            - self.parameters["lza"] * zo * (zo + self.parameters["zo0"])
            - self.parameters["lzd"] * zo * (zo + self.parameters["zo0"])
        )
        dde = (
            self.parameters["lpd"] * (p1 + p2 + p3)
            + self.parameters["lzd"] * zo * (zo + self.parameters["zo0"])
            - llda * de
        )
        dam = (
            llda * de
            - llan * am
            - am / (am + ni) * r1
            - am / (am + ni) * r2
            + self.parameters["lpa"] * (p1 + p2 + p3)
            + self.parameters["lza"] * zo * (zo + self.parameters["zo0"])
        )
        dni = llan * am - ni / (am + ni) * r1 - ni / (am + ni) * r2 - self.parameters["s1"] * llda * de * thomnp
        dpo = self.parameters["sr"] * (
            -r1
            - r2
            - r3
            + llda * de
            + self.parameters["lpa"] * (p1 + p2 + p3)
            + self.parameters["lza"] * zo * (zo + self.parameters["zo0"])
        )
        do2 = (
            self.parameters["s2"] * (am / (am + ni) * (r1 + r2) + r3)
            - self.parameters["s4"] * (llan * am)
            + self.parameters["s3"] * (ni / (am + ni) * (r1 + r2))
            - self.parameters["s2"] * (thopnp + thomnm) * llda * de
            - self.parameters["s2"]
            * (self.parameters["lpa"] * (p1 + p2 + p3) + self.parameters["lza"] * zo * (zo + self.parameters["zo0"]))
        )

        dx = torch.stack([dp1, dp2, dp3, dzo, dde, dam, dni, dpo, do2], axis=-2)

        if isinstance(concentration, TensorDict):
            return TensorDict({"pelagic": dx}, batch_size=concentration.batch_size)
        else:
            return dx

    def do_bottom(self, concentration: TensorDict, dynamic_inputs: TensorDict, *args, **kwargs) -> TensorDict:
        """Calculate right-hand side of biological time step ODE as a vector with temporal derivatives for bottom level.

        Args:
            concentration (TensorDict): The current system state. Should have the fields "pelagic" and "benthic".
                If this model is coupled with a light model, *concentration* should also
                have the fields "light_par" and "light_par0".
            dynamic_inputs (TensorDict): Time-dependent model inputs. Should have the fields "taub", "temp", "salt".
                If this model is not coupled to a light model, should also have the corresponding light fields.

        Returns:
            TensorDict: Vector of temporal derivatives
        """
        x = concentration["pelagic"]

        de_bottom = x[..., 4, 0]
        ni_bottom = x[..., 6, 0]
        o2_bottom = x[..., 8, 0]

        flf = concentration["benthic"][..., 0]

        temp_bottom = dynamic_inputs["temp"][..., 0]
        taub = dynamic_inputs["taub"]

        thopnp = self.th(o2_bottom, self.parameters["wo"], 0.0, 1.0) * self.yy(self.parameters["wn"], ni_bottom)
        thomnp = self.th(-o2_bottom, self.parameters["wo"], 0.0, 1.0) * self.yy(self.parameters["wn"], ni_bottom)
        thomnm = self.th(-o2_bottom, self.parameters["wo"], 0.0, 1.0) * (
            1.0 - self.yy(self.parameters["wn"], ni_bottom)
        )
        thsum = thopnp + thomnp + thomnm
        thopnp = thopnp / thsum
        thomnp = thomnp / thsum
        thomnm = thomnm / thsum

        llsa = (
            self.parameters["lsa"]
            * torch.exp(self.parameters["bsa"] * temp_bottom)
            * (self.th(o2_bottom, self.parameters["wo"], self.dot2, 1.0))
        )

        # Sedimentation dependent on bottom stress
        llds = torch.clamp_min(
            self.parameters["lds"] * (self.parameters["tau_crit"] - taub) / self.parameters["tau_crit"], 0.0
        )

        # Resuspension dependent on bottom stress
        llsd = torch.clamp_min(
            self.parameters["lsd"] * (taub - self.parameters["tau_crit"]) / self.parameters["tau_crit"], 0.0
        )

        dfl = (
            llds * de_bottom
            - llsd * flf
            - llsa * flf
            - self.th(o2_bottom, self.parameters["wo"], 0.0, 1.0) * llsa * flf
        )

        fl_de = -llds * de_bottom + llsd * flf
        fl_am = llsa * flf
        fl_ni = -self.parameters["s1"] * thomnp * llsa * flf
        fl_po = (
            self.parameters["sr"]
            * (
                1.0
                - self.parameters["ph1"]
                * self.th(o2_bottom, self.parameters["wo"], 0.0, 1.0)
                * self.yy(self.parameters["ph2"], o2_bottom)
            )
            * llsa
            * flf
        )
        fl_o2 = -(self.parameters["s4"] + self.parameters["s2"] * (thopnp + thomnm)) * llsa * flf
        fl_zero = torch.full_like(fl_o2, 0.0)

        fl_x = torch.stack([fl_zero, fl_zero, fl_zero, fl_zero, fl_de, fl_am, fl_ni, fl_po, fl_o2], axis=-1).unsqueeze(
            -1
        )

        d_benthic = dfl.unsqueeze(-1)

        return TensorDict({"pelagic": fl_x, "benthic": d_benthic}, batch_size=concentration.batch_size)

    def do_surface(
        self, concentration: Union[TensorDict, torch.Tensor], dynamic_inputs: TensorDict, *args, **kwargs
    ) -> Union[TensorDict, torch.Tensor]:
        """Calculate right-hand side of biological time step ODE as a vector with temporal derivatives at surface level.

        Args:
            concentration (Union[TensorDict, torch.Tensor]): The current system state.
                Should have the fields "pelagic" and "benthic".
                If this model is coupled with a light model, *concentration* should also
                have the fields "light_par" and "light_par0".
            dynamic_inputs (TensorDict): Time-dependent model inputs. Should have the fields "taub", "temp", "salt".
                If this model is not coupled to a light model, should also have the corresponding light fields.

        Returns:
            Union[TensorDict, torch.Tensor]: Vector of temporal derivatives
        """
        if isinstance(concentration, torch.Tensor):
            x = concentration
        elif isinstance(concentration, TensorDict):
            x = concentration["pelagic"]

        temp_surface = dynamic_inputs["temp"][..., -1]
        salt_surface = dynamic_inputs["salt"][..., -1]
        o2_surface = x[..., 8, -1]
        wind = dynamic_inputs["wind"]

        sc = 1450.0 + (1.1 * temp_surface - 71.0) * temp_surface
        p_vel = self.mask_func(
            wind - 13.0,
            5.9 * (5.9 * wind - 49.3) / torch.sqrt(sc),
            self.mask_func(3.6 - wind, 1.003 * wind / sc**0.66, 5.9 * (2.85 * wind - 9.65) / torch.sqrt(sc)),
        )
        p_vel = p_vel.clamp_min(0.05) / 86400.0

        fl_o2 = p_vel * (self.osat_weiss(temp_surface, salt_surface) - o2_surface)

        fl_ni = torch.full_like(fl_o2, self.parameters["sfl_ni"])
        fl_am = torch.full_like(fl_o2, self.parameters["sfl_am"])
        fl_po = torch.full_like(fl_o2, self.parameters["sfl_po"])
        fl_zero = torch.full_like(fl_o2, 0.0)

        fl_x = torch.stack(
            [fl_zero, fl_zero, fl_zero, fl_zero, fl_zero, fl_am, fl_ni, fl_po, fl_o2], axis=-1
        ).unsqueeze(-1)

        if isinstance(concentration, TensorDict):
            return TensorDict({"pelagic": fl_x}, batch_size=concentration.batch_size)
        else:
            return fl_x

    def do_ppdd(self):
        """Calculate source/sink terms of biological variables as production and destruction matrices.

        Raises:
            NotImplementedError: This method still has to be implemented.
        """
        raise NotImplementedError()

    # For the BioModel.right_hand_side_rhs method, we want to hide the do_bottom method in case self.fluff is False
    def __getattribute__(self, name):
        """Hide do_bottom if fluff is False."""
        if name == "do_bottom" and not object.__getattribute__(self, "fluff"):
            raise AttributeError("hidden")
        return object.__getattribute__(self, name)

    def __dir__(self):
        """Hide do_bottom from dir() if fluff is False."""
        d = super().__dir__()
        if not self.fluff and "do_bottom" in d:
            d.remove("do_bottom")
        return d
