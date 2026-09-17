from typing import Union

import torch
import yaml
from marinade.fabm.bio_model import BioModel
from tensordict import TensorDict


class NerscEcosmoSimpleModel(BioModel):
    """This class describes a simplified version of the biosystem model ECOSMO developed by the Nansen center.

    If created by the constructor method, this model uses the same default parameters as the version of this model
    provided by FABM. A model with custom parameters can be created from a .yaml file with
    *NerscEcosmoSimpleModel.create_model_from_yaml_file*.

    Args:
        ode_method (str): The numerical method for solving the tracer ODE. Supported schemes:
            ("FE": Forward Euler, "RK2": 2nd order Runge-Kutta, "RK4": 4th order Runge-Kutta)
        model_name (str, optional): The name of the model. Defaults to "ecosmo".
        device (str, optional): The device where the tensors of the model should be stored. Defaults to "cpu".
        dtype (torch.dtype, optional): The default data type of the model. Defaults to torch.float64.
    """

    def __init__(
        self,
        ode_method: str,
        model_name: str = "ecosmo",
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        *args,
        **kwargs,
    ):
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)

        super().__init__(ode_method=ode_method, model_name=model_name, device=device, dtype=dtype, *args, **kwargs)

        # define Redfield conversion ratios
        self.redf = torch.tensor(
            [
                6.625,  # C_N
                106.0,  # C_P
                6.625,  # C_SiO
                16.0,  # N_P
                1.0,  # N_SiO
                12.01,  # C_Cmg
                44.6608009,  # O2mm_ml
                14.007,  # N_Nmg
                30.97,  # P_Pmg
                28.09,  # Si_Simg
            ],
            device=device,
            dtype=dtype,
        )
        self.redf = torch.cat([self.redf, 1.0 / self.redf])

        self.parameters["zpr"] = 0.001 / 86400
        self.parameters["frr"] = 0.4
        # surface fluxes (in mgC/m2/s)
        self.parameters["surface_deposition_no3"] = 0.0 * self.redf[0] * self.redf[5] / 86400
        self.parameters["surface_deposition_nh4"] = 0.0 * self.redf[0] * self.redf[5] / 86400
        self.parameters["surface_deposition_pho"] = 0.0 * self.redf[1] * self.redf[5] / 86400

        self.parameters["muP"] = torch.tensor(1.2 / 86400, device=device, dtype=dtype)
        self.parameters["aa"] = torch.tensor(0.04, device=device, dtype=dtype)
        self.parameters["EXw"] = torch.tensor(0.041, device=device, dtype=dtype)
        self.parameters["EXphy"] = 0.03 / (self.redf[0] * self.redf[5])
        self.parameters["extdet"] = torch.tensor(0.0 / 1000.0, device=device, dtype=dtype)
        self.parameters["extdom"] = torch.tensor(0.0 / 1000.0, device=device, dtype=dtype)
        self.parameters["rNH4"] = 0.2 * self.redf[0] * self.redf[5]
        self.parameters["rNO3"] = 0.5 * self.redf[0] * self.redf[5]
        self.parameters["psi"] = 3.0 / (self.redf[0] * self.redf[5])
        self.parameters["mP"] = torch.tensor(0.06 / 86400, device=device, dtype=dtype)
        self.parameters["GrZP"] = torch.tensor(0.90 / 86400, device=device, dtype=dtype)
        self.parameters["Rg"] = 0.50 * self.redf[0] * self.redf[5]
        self.parameters["mZ"] = torch.tensor(0.15 / 86400, device=device, dtype=dtype)
        self.parameters["excZ"] = torch.tensor(0.07 / 86400, device=device, dtype=dtype)
        self.parameters["gammaZp"] = torch.tensor(0.75, device=device, dtype=dtype)
        self.parameters["gammaZd"] = torch.tensor(0.75, device=device, dtype=dtype)
        self.parameters["reminD"] = torch.tensor(0.003 / 86400, device=device, dtype=dtype)
        self.parameters["sinkDet"] = torch.tensor(5.00 / 86400, device=device, dtype=dtype)
        self.parameters["rPO4"] = 0.05 * self.redf[1] * self.redf[5]
        self.parameters["crBotStr"] = torch.tensor(0.007, device=device, dtype=dtype)
        self.parameters["resuspRt"] = torch.tensor(25.0 / 86400, device=device, dtype=dtype)
        self.parameters["sedimRt"] = torch.tensor(3.5 / 86400, device=device, dtype=dtype)
        self.parameters["burialRt"] = torch.tensor(1e-5 / 86400, device=device, dtype=dtype)
        self.parameters["reminSED"] = torch.tensor(0.001 / 86400, device=device, dtype=dtype)
        self.parameters["TctrlDenit"] = torch.tensor(0.15, device=device, dtype=dtype)
        self.parameters["RelSEDp1"] = torch.tensor(0.15, device=device, dtype=dtype)
        self.parameters["RelSEDp2"] = torch.tensor(0.10, device=device, dtype=dtype)

        #  growth fractions
        self.parameters["prefZp"] = torch.tensor(0.70, device=device, dtype=dtype)
        self.parameters["prefZD"] = torch.tensor(0.00, device=device, dtype=dtype)

        self.tracer_names.extend(
            [
                f"{self.model_name}_no3",
                f"{self.model_name}_nh4",
                f"{self.model_name}_pho",
                f"{self.model_name}_oxy",
                f"{self.model_name}_phy",
                f"{self.model_name}_zoo",
                f"{self.model_name}_det",
                f"{self.model_name}_dom",
            ]
        )
        self.benthic_tracer_names.extend(
            [
                f"{self.model_name}_sed1",
                f"{self.model_name}_sed3",
            ]
        )

        self.dependencies.extend(["temp", "salt", "taub"])

        self.setup_internal_fields()

    def setup_internal_fields(self):
        """Routine for setting up internal fields that depend on internal parameters."""
        device = self.redf.device
        dtype = self.redf.dtype

        _ws = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -self.parameters["sinkDet"], 0.0]
        self.ws = torch.tensor(_ws, device=device, dtype=dtype).unsqueeze(0).unsqueeze(-1)

        _minimum = [0.0, 0.0, 0.0, 0.0, 1.0e-7, 1.0e-7, 0.0, 0.0]
        self.minimum = torch.tensor(_minimum, device=device, dtype=dtype).unsqueeze(0).unsqueeze(-1)

        self.minimum_td = TensorDict({"pelagic": self.minimum, "benthic": torch.zeros((2), dtype=dtype, device=device)})

        light_scale_list = [self.parameters["EXphy"], self.parameters["extdet"], self.parameters["extdom"]]
        self.light_attenuation_indexes = [4, 6, 7]
        self.light_attenuation_scale_factor = torch.stack(light_scale_list).unsqueeze(-1)
        self.light_attenuation_background = self.parameters["EXw"]

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
        """Creates a NerscEcosmoSimpleModel instance from .yaml configuration files. Should generally be used instead of
        the constructor. Designed to be compatible with the configuration files of GOTM.

        Args:
            path_fabm (str): Path to the FABM .yaml configuration file
            path_gotm (str): Path to the GOTM .yaml configuration file
            device (str, optional): Torch device. Defaults to "cpu".
            dtype (Union[str, torch.dtype], optional): Torch data type. Defaults to torch.float64.

        Returns:
            NerscEcosmoSimpleModel: The generated model
        """

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

        model = None

        for name_instance, config_instance in config_fabm["instances"].items():
            if config_instance["model"].replace("/", "_") == "nersc_ecosmo_simple":
                if model is not None:
                    raise ValueError(f"More than one Ecosmo operational model detected in file {file_fabm}!")

                model = cls(
                    ode_method=ode_method,
                    model_name=name_instance,
                    device=device,
                    dtype=dtype,
                    repair_state=repair_state,
                    *args,
                    **kwargs,
                )

                config_par = config_instance["parameters"]

                model.parameters["zpr"] = torch.tensor(config_par["zpr"] / 86400, device=device, dtype=dtype)
                model.parameters["frr"] = torch.tensor(config_par["frr"], device=device, dtype=dtype)
                # surface fluxes (in mgC/m2/s)
                model.parameters["surface_deposition_no3"] = (
                    config_par["surface_deposition_no3"] * model.redf[0] * model.redf[5] / 86400
                )
                model.parameters["surface_deposition_nh4"] = (
                    config_par["surface_deposition_nh4"] * model.redf[0] * model.redf[5] / 86400
                )
                model.parameters["surface_deposition_pho"] = (
                    config_par["surface_deposition_pho"] * model.redf[1] * model.redf[5] / 86400
                )

                model.parameters["muP"] = torch.tensor(config_par["muP"] / 86400, device=device, dtype=dtype)
                model.parameters["aa"] = torch.tensor(config_par["aa"], device=device, dtype=dtype)
                model.parameters["EXw"] = torch.tensor(config_par["EXw"], device=device, dtype=dtype)
                model.parameters["EXphy"] = config_par["Exphy"] / (model.redf[0] * model.redf[5])
                model.parameters["rNH4"] = config_par["rNH4"] * model.redf[0] * model.redf[5]
                model.parameters["rNO3"] = config_par["rNO3"] * model.redf[0] * model.redf[5]
                model.parameters["psi"] = config_par["psi"] / (model.redf[0] * model.redf[5])
                model.parameters["mP"] = torch.tensor(config_par["mP"] / 86400, device=device, dtype=dtype)
                model.parameters["GrZP"] = torch.tensor(config_par["GrZP"] / 86400, device=device, dtype=dtype)
                model.parameters["Rg"] = config_par["Rg"] * model.redf[0] * model.redf[5]
                model.parameters["mZ"] = torch.tensor(config_par["mZ"] / 86400, device=device, dtype=dtype)
                model.parameters["excZ"] = torch.tensor(config_par["excZ"] / 86400, device=device, dtype=dtype)
                model.parameters["gammaZp"] = torch.tensor(config_par["gammaZp"], device=device, dtype=dtype)
                model.parameters["gammaZd"] = torch.tensor(config_par["gammaZd"], device=device, dtype=dtype)
                model.parameters["reminD"] = torch.tensor(config_par["reminD"] / 86400, device=device, dtype=dtype)
                model.parameters["sinkDet"] = torch.tensor(config_par["sinkDet"] / 86400, device=device, dtype=dtype)
                model.parameters["rPO4"] = config_par["rPO4"] * model.redf[1] * model.redf[5]
                model.parameters["crBotStr"] = torch.tensor(config_par["crBotStr"], device=device, dtype=dtype)
                model.parameters["resuspRt"] = torch.tensor(config_par["resuspRt"] / 86400, device=device, dtype=dtype)
                model.parameters["sedimRt"] = torch.tensor(config_par["sedimRt"] / 86400, device=device, dtype=dtype)
                model.parameters["burialRt"] = torch.tensor(config_par["burialRt"] / 86400, device=device, dtype=dtype)
                model.parameters["reminSED"] = torch.tensor(config_par["reminSED"] / 86400, device=device, dtype=dtype)
                model.parameters["TctrlDenit"] = torch.tensor(config_par["TctrlDenit"], device=device, dtype=dtype)
                model.parameters["RelSEDp1"] = torch.tensor(config_par["RelSEDp1"], device=device, dtype=dtype)
                model.parameters["RelSEDp2"] = torch.tensor(config_par["RelSEDp2"], device=device, dtype=dtype)

        model.parameters["extdet"] = torch.tensor(config_par["Exdet"] / 1000, device=device, dtype=dtype)
        model.parameters["extdom"] = torch.tensor(config_par["Exdom"] / 1000, device=device, dtype=dtype)

        #  growth fractions
        model.parameters["prefZp"] = torch.tensor(config_par["prefZP"], device=device, dtype=dtype)
        model.parameters["prefZD"] = torch.tensor(config_par["prefZD"], device=device, dtype=dtype)

        model.setup_internal_fields()

        return model

    def update_parameters(self, parameters: TensorDict):
        """Update the parameters of the model.

        Args:
            parameters (TensorDict): The model parameters to be updated
        """
        super().update_parameters(parameters)

        if "sinkDet" in parameters:
            _zero = torch.zeros_like(self.parameters["sinkDet"])
            _ws = [_zero, _zero, _zero, _zero, _zero, _zero, -self.parameters["sinkDet"], _zero]
            self.ws = torch.stack(_ws).unsqueeze(0).unsqueeze(-1)

        if any(k in parameters for k in ("EXphy", "extdet", "extdom")):
            light_scale_list = [self.parameters["EXphy"], self.parameters["extdet"], self.parameters["extdom"]]
            self.light_attenuation_scale_factor = torch.stack(light_scale_list).unsqueeze(-1)
        if "EXw" in parameters:
            self.light_attenuation_background = self.parameters["EXw"]

    def do(self, concentration: TensorDict, dynamic_inputs: TensorDict, *args, **kwargs) -> TensorDict:
        """Calculate right-hand side of biological time step ODE as a vector with temporal derivatives.

        Args:
            concentration (TensorDict): The current system state. Should have the fields "pelagic" and "benthic".
                If this model is coupled with a light model, *concentration* should also
                have the fields "light_par", "light_par0" and also "light_par_mean" and "light_par0_mean"
                for the temporal mean values, depending on the mode of time averaging in the light model.
                The time averages can also be provided in *dynamic_inputs*.
            dynamic_inputs (TensorDict): Time-dependent model inputs. Should have the fields "taub", "temp", "salt".
                If this model is not coupled to a light model, should also have the corresponding light fields "par".

        Returns:
            TensorDict: Vector of temporal derivatives
        """

        use_state_light = False

        if isinstance(concentration, torch.Tensor):
            x = concentration
        elif isinstance(concentration, TensorDict):
            x = concentration["pelagic"]
            use_state_light = "light_par" in concentration

        no3 = x[..., 0, :]
        nh4 = x[..., 1, :]
        pho = x[..., 2, :]
        oxy = x[..., 3, :]
        phy = x[..., 4, :]
        zoo = x[..., 5, :]
        det = x[..., 6, :]
        dom = x[..., 7, :]

        if use_state_light:
            par = concentration["light_par"]
        else:
            par = dynamic_inputs["par"]

        temp = dynamic_inputs["temp"]

        zero = torch.tensor(0.0).to(phy)
        one = torch.tensor(1.0).to(phy)

        phy_loss = self.mask_func(0.01 - phy, zero, one)
        zoo_loss = self.mask_func(0.001 - zoo, zero, one)

        # remineralisation rate
        frem = self.parameters["reminD"] * (1.0 + 20.0 * (temp**2 / (13.0**2 + temp**2)))
        fremDOM = 10.0 * frem

        # nutrient limitation factors
        # k denotes half-saturation values
        up_nh4 = nh4 / (self.parameters["rNH4"] + nh4)
        up_no3 = no3 / (self.parameters["rNO3"] + no3) * torch.exp(-self.parameters["psi"] * nh4)
        up_n = up_nh4 + up_no3
        up_pho = pho / (self.parameters["rPO4"] + pho)

        # light limitation - old formulation - default for now - CAGLAR:Sep2022
        blight = torch.clamp_min(torch.tanh(self.parameters["aa"] * par), 0.0)

        # temperature dependence
        Tphy = 1.0

        # production and nutrient uptake
        P_prod = Tphy * torch.minimum(blight, torch.minimum(up_n, up_pho))
        Prod = self.parameters["muP"] * P_prod * phy

        # grazing
        Fs = self.parameters["prefZp"] * phy + self.parameters["prefZD"] * det

        ZonP = phy_loss * self.parameters["GrZP"] * self.parameters["prefZp"] * phy / (self.parameters["Rg"] + Fs)
        ZonD = self.parameters["GrZP"] * self.parameters["prefZD"] * det / (self.parameters["Rg"] + Fs)

        # nitrification
        Onitr = 0.01 * self.redf[6]  # according to Neumann  (Onitr in mlO2/l see also Stigebrand and Wulff)
        bioom1 = self.mask_func(oxy, 0.1 / 86400 * torch.exp(temp * 0.11) * oxy / (Onitr + oxy), zero)
        bioom3 = zero
        bioom5 = self.mask_func(no3, self.mask_func(oxy, zero, 5.0), zero)
        bioom6 = self.mask_func(oxy, one, zero)
        bioom7 = self.mask_func(no3, zero, self.mask_func(oxy, zero, one))

        d_phy = (self.parameters["muP"] * P_prod - self.parameters["mP"] * phy_loss) * phy - ZonP * zoo

        # microzooplankton
        Z_prod = self.parameters["gammaZp"] * ZonP + self.parameters["gammaZd"] * ZonD
        d_zoo = (Z_prod - (self.parameters["mZ"] + self.parameters["excZ"] + self.parameters["zpr"]) * zoo_loss) * zoo

        # detritus
        dxxdet = (
            ((one - self.parameters["gammaZp"]) * ZonP + (one - self.parameters["gammaZd"]) * ZonD) * zoo
            + self.parameters["mZ"] * zoo * zoo_loss
            + self.parameters["mP"] * phy * phy_loss
        )

        d_det = (1.0 - self.parameters["frr"]) * dxxdet - ZonD * zoo - frem * det

        # labile dissolved organic matter
        d_dom = self.parameters["frr"] * dxxdet - fremDOM * dom

        # nitrate
        rhs_nit = (
            -(up_no3 + 0.5e-10) / (up_n + 1.0e-10) * Prod
            + bioom1 * nh4
            - bioom3 * no3
            - frem * det * bioom5
            - fremDOM * dom * bioom5
        )
        d_no3 = rhs_nit

        # ammonium
        rhs_amm = (
            -(up_nh4 + 0.5e-10) / (up_n + 1.0e-10) * Prod
            + self.parameters["excZ"] * zoo * zoo_loss
            + frem * det
            + fremDOM * dom
            - bioom1 * nh4
        )
        d_nh4 = rhs_amm

        # phosphate
        rhs_pho = -Prod + self.parameters["excZ"] * zoo * zoo_loss + frem * det + fremDOM * dom
        d_pho = rhs_pho

        # oxygen
        rhs_oxy = (
            (
                (6.625 * up_nh4 + 8.125 * up_no3 + 1.0e-10) / (up_n + 1.0e-10) * Prod
                - bioom6 * 6.625 * (self.parameters["excZ"] * zoo * zoo_loss)
                - frem * det * (bioom6 + bioom7) * 6.625
                - (bioom6 + bioom7) * 6.625 * fremDOM * dom
                - 2.0 * bioom1 * nh4
            )
            * self.redf[10]
            * self.redf[15]
        )
        d_oxy = rhs_oxy

        dx_list = [d_no3, d_nh4, d_pho, d_oxy, d_phy, d_zoo, d_det, d_dom]
        dx = torch.stack(dx_list, axis=-2)

        if isinstance(concentration, TensorDict):
            return TensorDict({"pelagic": dx}, batch_size=concentration.batch_size)
        else:
            return dx

    def do_bottom(self, concentration: TensorDict, dynamic_inputs: TensorDict, *args, **kwargs) -> TensorDict:
        """Calculate right-hand side of biological time step ODE as a vector with temporal derivatives for bottom level.

        Args:
            concentration (TensorDict): The current system state. Should have the fields "pelagic" and "benthic".
                If this model is coupled with a light model, _concentration_ should also have the fields
                "light_par" and "light_par0".
            dynamic_inputs (TensorDict): Time-dependent model inputs. Should have the fields "taub", "temp", "salt".
                If this model is not coupled to a light model, should also have the corresponding light fields.

        Returns:
            TensorDict: Vector of temporal derivatives
        """
        x = concentration["pelagic"]

        no3_bottom = x[..., 0, 0]
        oxy_bottom = x[..., 3, 0]
        det_bottom = x[..., 6, 0]

        sed1 = concentration["benthic"][..., 0]
        sed3 = concentration["benthic"][..., 1]

        temp_bottom = dynamic_inputs["temp"][..., 0]
        taub = dynamic_inputs["taub"]

        zero = torch.tensor(0.0).to(no3_bottom)
        one = torch.tensor(1.0).to(no3_bottom)

        bioom1 = self.mask_func(
            oxy_bottom,
            0.1 / 86400 * torch.exp(temp_bottom * 0.11) * oxy_bottom / (0.1 * self.redf[6] + oxy_bottom),
            0.0,
        )
        bioom5 = self.mask_func(no3_bottom, self.mask_func(oxy_bottom, zero, 5.0), zero)
        bioom6 = self.mask_func(oxy_bottom, one, zero)
        bioom7 = self.mask_func(no3_bottom, zero, self.mask_func(oxy_bottom, zero, one))

        # ----citical bottom shear stress
        Rsd = self.mask_func(
            self.parameters["crBotStr"] - taub,
            0.0,
            torch.clamp_max(self.parameters["resuspRt"] * taub**3 / (0.1 + taub**3), self.parameters["resuspRt"]),
        )
        Rds = self.mask_func(self.parameters["crBotStr"] - taub, self.parameters["sedimRt"], zero)

        # ---------------------------------------------------------------
        # ----denitrification parameter in dependence of available oxygen
        Rsa = self.mask_func(
            oxy_bottom, self.parameters["reminSED"] * torch.exp(self.parameters["TctrlDenit"] * temp_bottom), zero
        )
        Rsdenit = self.mask_func(
            oxy_bottom, zero, 2.0 * self.parameters["reminSED"] * torch.exp(self.parameters["TctrlDenit"] * temp_bottom)
        )

        # --- sediment 1 total sediment biomass and nitrogen pool
        rhs = Rds * det_bottom - Rsd * sed1 - 2.0 * Rsa * sed1 - Rsdenit * sed1 - self.parameters["burialRt"] * sed1
        d_sed1 = rhs

        # oxygen
        flux = (
            -(bioom6 * 6.625 * 2.0 * Rsa * sed1 + bioom7 * 6.625 * Rsdenit * sed1 + 2.0 * bioom1 * Rsa * sed1)
            * self.redf[10]
            * self.redf[15]
        )
        fl_oxy = flux

        # nitrate
        fl_no3 = -bioom5 * Rsdenit * sed1

        # detritus
        fl_det = Rsd * sed1 - Rds * det_bottom

        # ammonium
        fl_nh4 = (Rsdenit + Rsa) * sed1

        # --try out for phosphate ute 2.6.2010
        Rsa_p = 2.0 * self.parameters["reminSED"] * torch.exp(self.parameters["TctrlDenit"] * temp_bottom)

        yt2 = oxy_bottom / 375.0  # normalize the value as in Neumann et al 2002
        yt1 = yt2**2 / (self.parameters["RelSEDp2"] ** 2 + yt2**2)

        fl_pho = self.mask_func(oxy_bottom, Rsa_p * (one - self.parameters["RelSEDp1"] * yt1) * sed3, Rsa_p * sed3)

        # --sed 3 phosphate pool sediment+remineralization-P release
        d_sed3 = self.mask_func(
            oxy_bottom,
            2.0 * Rsa * sed1 - Rsa_p * (one - self.parameters["RelSEDp1"] * yt1) * sed3,
            Rsdenit * sed1 - Rsa_p * sed3,
        )

        fl_zero = torch.full_like(fl_no3, zero)

        fl_x_list = [fl_no3, fl_nh4, fl_pho, fl_oxy, fl_zero, fl_zero, fl_det, fl_zero]
        fl_x = torch.stack(fl_x_list, axis=-1).unsqueeze(-1)

        d_benthic = torch.stack([d_sed1, d_sed3], axis=-1)

        return TensorDict({"pelagic": fl_x, "benthic": d_benthic}, batch_size=concentration.batch_size)

    def do_surface(self, concentration: TensorDict, dynamic_inputs: TensorDict, *args, **kwargs) -> TensorDict:
        """Calculate right-hand side of biological time step ODE as a vector with temporal derivatives for surface
        level.

        Args:
            concentration (TensorDict): The current system state. Should have the fields "pelagic" and "benthic".
                If this model is coupled with a light model, _concentration_ should also
                have the fields "light_par" and "light_par0".
            dynamic_inputs (TensorDict): Time-dependent model inputs. Should have the fields "taub", "temp", "salt".
                If this model is not coupled to a light model, should also have the corresponding light fields.

        Returns:
            TensorDict: Vector of temporal derivatives
        """

        if isinstance(concentration, torch.Tensor):
            x = concentration
        elif isinstance(concentration, TensorDict):
            x = concentration["pelagic"]

        temp_surface = dynamic_inputs["temp"][..., -1]
        salt_surface = dynamic_inputs["salt"][..., -1]

        oxy_surface = x[..., 3, -1]

        # Oxygen saturation micromol/liter__(Benson and Krause, 1984)
        tr = 1.0 / (temp_surface + 273.15)
        o2sat = torch.exp(
            -135.90205
            + (1.575701e05) * tr
            - (6.642308e07) * tr**2
            + (1.243800e10) * tr**3
            - (8.621949e11) * tr**4
            - salt_surface * (0.017674 - 10.754 * tr + 2140.7 * tr**2)
        )

        fl_oxy = 5.0 / 86400.0 * (o2sat - oxy_surface)
        fl_no3 = torch.full_like(fl_oxy, self.parameters["surface_deposition_no3"])
        fl_nh4 = torch.full_like(fl_oxy, self.parameters["surface_deposition_nh4"])
        fl_pho = torch.full_like(fl_oxy, self.parameters["surface_deposition_pho"])
        fl_zero = torch.full_like(fl_oxy, 0.0)

        fl_x_list = [fl_no3, fl_nh4, fl_pho, fl_oxy, fl_zero, fl_zero, fl_zero, fl_zero]

        fl_x = torch.stack(fl_x_list, axis=-1).unsqueeze(-1)

        if isinstance(concentration, TensorDict):
            return TensorDict({"pelagic": fl_x}, batch_size=concentration.batch_size)
        else:
            return fl_x
