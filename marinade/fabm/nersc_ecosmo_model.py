from typing import Union

import torch
import yaml
from marinade.fabm.bio_model import BioModel
from tensordict import TensorDict


class NerscEcosmoModel(BioModel):
    """This class describes the biosystem model ECOSMO developed by the Nansen center and Hereon.

    If created by the constructor method, this model uses the same default parameters as the version of this model
    provided by FABM. A model with custom parameters can be created from a .yaml file with
    *NerscEcosmoModel.create_model_from_yaml_file*.

    Args:
        use_cyanos (bool): Whether to include cyanobacteria as a tracer in the model or not
        use_fish (bool): Whether to include fish and macrobenthos in the model.
        couple_co2 (bool): Whether to couple this model with a CO2 model. Currently not supported.
        use_chl (bool): Whether to use additional tracers associated with chlorophyll.
        ode_method (str): The numerical method for solving the tracer ODE. Supported schemes:
            ("FE": Forward Euler, "RK2": 2nd order Runge-Kutta, "RK4": 4th order Runge-Kutta)
        turn_on_additional_diagnostics (bool, optional): Legacy parameter that has no effect. Defaults to False.
        use_corrected_bioc (bool, optional): Whether to use a suggested correction to the model concerning the
            parameter "Exphy" or "phyto self-shading". If this is True, the parameter depends on *use_chl*.
            Defaults to False.
        model_name (str, optional): The name of the model. Defaults to "ecosmo".
        device (str, optional): The device where the tensors of the model should be stored. Defaults to "cpu".
        dtype (torch.dtype, optional): The default data type of the model. Defaults to torch.float64.
    """

    def __init__(
        self,
        use_cyanos: bool,
        use_fish: bool,
        couple_co2: bool,
        use_chl: bool,
        community_export: bool,
        ode_method: str,
        turn_on_additional_diagnostics: bool = False,
        use_corrected_bioc: bool = False,
        model_name: str = "ecosmo",
        device="cpu",
        dtype: torch.dtype = torch.float64,
        *args,
        **kwargs,
    ):
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)

        super().__init__(ode_method=ode_method, model_name=model_name, device=device, dtype=dtype, *args, **kwargs)

        self.use_cyanos = use_cyanos
        self.use_fish = use_fish
        if use_fish:
            raise NotImplementedError(
                "The parameter use_fish for NerscEcosmoModel is currently not supported. It must be set to False."
            )
        if couple_co2:
            raise NotImplementedError(
                "The parameter couple_co2 for NerscEcosmoModel is currently not supported. It must be set to False."
            )
        self.couple_co2 = couple_co2
        self.use_chl = use_chl
        self.community_export = community_export
        if community_export:
            raise NotImplementedError(
                "The parameter community_export for NerscEcosmoModel is currently not supported."
                "It must be set to False."
            )
        self.turn_on_additional_diagnostics = turn_on_additional_diagnostics

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

        self.parameters["zpr"] = torch.tensor(0.001 / 86400, device=device, dtype=dtype)
        self.parameters["frr"] = torch.tensor(0.4, device=device, dtype=dtype)
        self.parameters["nfixation_minimum_daily_par"] = torch.tensor(40.0, device=device, dtype=dtype)
        self.parameters["bg_growth_minimum_daily_rad"] = torch.tensor(120.0, device=device, dtype=dtype)
        # surface fluxes (in mgC/m2/s)
        self.parameters["surface_deposition_no3"] = 0.0 * self.redf[0] * self.redf[5] / 86400
        self.parameters["surface_deposition_nh4"] = 0.0 * self.redf[0] * self.redf[5] / 86400
        self.parameters["surface_deposition_pho"] = 0.0 * self.redf[1] * self.redf[5] / 86400
        self.parameters["surface_deposition_sil"] = 0.0 * self.redf[2] * self.redf[5] / 86400

        # parameters stored in the array bio_c in fortran
        self.parameters["muPl"] = torch.tensor(1.3 / 86400, device=device, dtype=dtype)
        self.parameters["muPs"] = torch.tensor(1.1 / 86400, device=device, dtype=dtype)
        self.parameters["aa"] = torch.tensor(0.04, device=device, dtype=dtype)
        self.parameters["EXw"] = torch.tensor(0.041, device=device, dtype=dtype)
        self.parameters["Exphy"] = (
            torch.tensor(0.04, device=device, dtype=dtype)
            if (self.use_chl and use_corrected_bioc)
            else (0.04 / (self.redf[0] * self.redf[5]))
        )
        self.parameters["rNH4"] = 0.2 * self.redf[0] * self.redf[5]
        self.parameters["rNO3"] = 0.5 * self.redf[0] * self.redf[5]
        self.parameters["psi"] = 3.0 / (self.redf[0] * self.redf[5])
        self.parameters["mPl"] = torch.tensor(0.04 / 86400, device=device, dtype=dtype)
        self.parameters["mPs"] = torch.tensor(0.08 / 86400, device=device, dtype=dtype)
        self.parameters["GrZlP"] = torch.tensor(0.80 / 86400, device=device, dtype=dtype)
        self.parameters["GrZsP"] = torch.tensor(1.00 / 86400, device=device, dtype=dtype)
        self.parameters["GrZlZ"] = torch.tensor(0.50 / 86400, device=device, dtype=dtype)
        self.parameters["Rg"] = 0.50 * self.redf[0] * self.redf[5]
        self.parameters["mZl"] = torch.tensor(0.10 / 86400, device=device, dtype=dtype)
        self.parameters["mZs"] = torch.tensor(0.20 / 86400, device=device, dtype=dtype)
        self.parameters["excZl"] = torch.tensor(0.06 / 86400, device=device, dtype=dtype)
        self.parameters["excZs"] = torch.tensor(0.08 / 86400, device=device, dtype=dtype)
        self.parameters["gammaZlp"] = torch.tensor(0.75, device=device, dtype=dtype)
        self.parameters["gammaZsp"] = torch.tensor(0.75, device=device, dtype=dtype)
        self.parameters["gammaZd"] = torch.tensor(0.75, device=device, dtype=dtype)
        self.parameters["reminD"] = torch.tensor(0.003 / 86400, device=device, dtype=dtype)
        self.parameters["sinkDet"] = torch.tensor(5.00 / 86400, device=device, dtype=dtype)
        self.parameters["Wa"] = torch.tensor(1.00 / 86400, device=device, dtype=dtype)
        self.parameters["rPO4"] = 0.05 * self.redf[1] * self.redf[5]
        self.parameters["rSi"] = 0.50 * self.redf[2] * self.redf[5]
        self.parameters["regenSi"] = torch.tensor(0.015 / 86400, device=device, dtype=dtype)
        self.parameters["muBG"] = torch.tensor(1.00 / 86400, device=device, dtype=dtype)
        self.parameters["TctrlBG"] = torch.tensor(1.00, device=device, dtype=dtype)
        self.parameters["TrefBG"] = torch.tensor(0.00, device=device, dtype=dtype)
        self.parameters["GrBG"] = torch.tensor(0.30 / 86400, device=device, dtype=dtype)
        self.parameters["mBG"] = torch.tensor(0.08 / 86400, device=device, dtype=dtype)
        self.parameters["upliftBG"] = torch.tensor(0.10 / 86400, device=device, dtype=dtype)
        self.parameters["crBotStr"] = torch.tensor(0.007, device=device, dtype=dtype)
        self.parameters["resuspRt"] = torch.tensor(25.0 / 86400, device=device, dtype=dtype)  # resuspRt
        self.parameters["sedimRt"] = torch.tensor(3.5 / 86400, device=device, dtype=dtype)  # sedimRt
        self.parameters["burialRt"] = torch.tensor(1e-5 / 86400, device=device, dtype=dtype)  # burialRt
        self.parameters["reminSED"] = torch.tensor(0.001 / 86400, device=device, dtype=dtype)  # reminSED
        self.parameters["TctrlDenit"] = torch.tensor(0.15, device=device, dtype=dtype)  # TctrlDenit
        self.parameters["RelSEDp1"] = torch.tensor(0.15, device=device, dtype=dtype)  # RelSEDp1
        self.parameters["RelSEDp2"] = torch.tensor(0.10, device=device, dtype=dtype)  # RelSEDp2
        self.parameters["reminSEDsi"] = torch.tensor(0.0002 / 86400, device=device, dtype=dtype)  # reminSEDsi
        self.parameters["sinkOPAL"] = torch.tensor(5.0 / 86400, device=device, dtype=dtype)  # sinkOPAL
        self.parameters["sinkBG"] = torch.tensor(-1.0 / 86400, device=device, dtype=dtype)  # sinkBG
        self.parameters["sinkDia"] = torch.tensor(0.0 / 86400, device=device, dtype=dtype)  # sinkDia

        self.parameters["extdet"] = torch.tensor(0.0 / 1000, device=device, dtype=dtype)
        self.parameters["extdom"] = torch.tensor(0.0 / 1000, device=device, dtype=dtype)

        #  growth fractions
        self.parameters["prefZsPs"] = torch.tensor(0.70, device=device, dtype=dtype)
        self.parameters["prefZsPl"] = torch.tensor(0.25, device=device, dtype=dtype)
        self.parameters["prefZsD"] = torch.tensor(0.00, device=device, dtype=dtype)
        self.parameters["prefZsBG"] = torch.tensor(0.30, device=device, dtype=dtype)
        self.parameters["prefZlPs"] = torch.tensor(0.10, device=device, dtype=dtype)
        self.parameters["prefZlPl"] = torch.tensor(0.85, device=device, dtype=dtype)
        self.parameters["prefZlZs"] = torch.tensor(0.15, device=device, dtype=dtype)
        self.parameters["prefZlD"] = torch.tensor(0.00, device=device, dtype=dtype)
        self.parameters["prefZlBG"] = torch.tensor(0.30, device=device, dtype=dtype)

        if self.use_fish:
            # Macrobenthos parameter settings by Ute
            self.parameters["mMB1"] = torch.tensor(0.001 / 86400, device=device, dtype=dtype)
            self.parameters["excMB1"] = torch.tensor(0.025 / 86400, device=device, dtype=dtype)
            self.parameters["rMB1"] = 0.5 * self.redf[0] * self.redf[5]
            self.parameters["GrMB1Z"] = torch.tensor(0.1 / 86400, device=device, dtype=dtype)
            self.parameters["GrMB1P"] = torch.tensor(0.1 / 86400, device=device, dtype=dtype)
            self.parameters["GrMB1Det"] = torch.tensor(0.1 / 86400, device=device, dtype=dtype)
            self.parameters["GrMB1Sed"] = torch.tensor(0.1 / 86400, device=device, dtype=dtype)
            self.parameters["tempcMB1"] = torch.tensor(0.0, device=device, dtype=dtype)
            self.parameters["prefMB1Zs"] = torch.tensor(0.2, device=device, dtype=dtype)
            self.parameters["prefMB1Zl"] = torch.tensor(0.3, device=device, dtype=dtype)
            self.parameters["prefMB1Det"] = torch.tensor(0.1, device=device, dtype=dtype)
            self.parameters["prefMB1Sed"] = torch.tensor(0.1, device=device, dtype=dtype)
            self.parameters["prefMB1P"] = torch.tensor(0.2, device=device, dtype=dtype)
            self.parameters["prefMB1Dom"] = torch.tensor(0.1, device=device, dtype=dtype)

            # parameter for fish1 functional group
            self.parameters["mF1"] = torch.tensor(0.001 / 86400, device=device, dtype=dtype)
            self.parameters["excF1"] = torch.tensor(0.002 / 86400, device=device, dtype=dtype)
            self.parameters["rF1"] = 0.70 * self.redf[0] * self.redf[5]
            self.parameters["rF1MB"] = 0.90 * self.redf[0] * self.redf[5]
            self.parameters["tempcF1"] = torch.tensor(0.5, device=device, dtype=dtype)
            self.parameters["asefF1"] = torch.tensor(0.7, device=device, dtype=dtype)
            self.parameters["GrF1Zl"] = torch.tensor(0.01 / 86400, device=device, dtype=dtype)
            self.parameters["GrF1Zs"] = torch.tensor(0.01 / 86400, device=device, dtype=dtype)
            self.parameters["GrF1Det"] = torch.tensor(0.05 / 86400, device=device, dtype=dtype)
            self.parameters["GrF1MB1"] = torch.tensor(0.01 / 86400, device=device, dtype=dtype)
            self.parameters["prefF1Zl"] = torch.tensor(0.45, device=device, dtype=dtype)
            self.parameters["prefF1Zs"] = torch.tensor(0.25, device=device, dtype=dtype)
            self.parameters["prefF1Det"] = torch.tensor(0.05, device=device, dtype=dtype)
            self.parameters["prefF1MB1"] = torch.tensor(0.25, device=device, dtype=dtype)
            # parameter for fish2 functional group
            self.parameters["mF2"] = torch.tensor(0.001 / 86400, device=device, dtype=dtype)
            self.parameters["excF2"] = torch.tensor(0.002 / 86400, device=device, dtype=dtype)
            self.parameters["rF2"] = 0.70 * self.redf[0] * self.redf[5]
            self.parameters["rF2MB"] = 0.90 * self.redf[0] * self.redf[5]
            self.parameters["tempcF2"] = torch.tensor(0.5, device=device, dtype=dtype)
            self.parameters["asefF2"] = torch.tensor(0.7, device=device, dtype=dtype)
            self.parameters["GrF2Zl"] = torch.tensor(0.01 / 86400, device=device, dtype=dtype)
            self.parameters["GrF2F1"] = torch.tensor(0.01 / 86400, device=device, dtype=dtype)
            self.parameters["GrF2Det"] = torch.tensor(0.05 / 86400, device=device, dtype=dtype)
            self.parameters["GrF2MB1"] = torch.tensor(0.01 / 86400, device=device, dtype=dtype)
            self.parameters["prefF2Zl"] = torch.tensor(0.45, device=device, dtype=dtype)
            self.parameters["prefF2F1"] = torch.tensor(0.25, device=device, dtype=dtype)
            self.parameters["prefF2Det"] = torch.tensor(0.05, device=device, dtype=dtype)
            self.parameters["prefF2MB1"] = torch.tensor(0.25, device=device, dtype=dtype)

        # chlorophyll-a constants
        self.parameters["MINchl2nPs"] = 0.50 * self.redf[10] * self.redf[15]
        self.parameters["MAXchl2nPs"] = 3.83 * self.redf[10] * self.redf[15]
        self.parameters["MINchl2nPl"] = 0.50 * self.redf[10] * self.redf[15]
        self.parameters["MAXchl2nPl"] = 2.94 * self.redf[10] * self.redf[15]
        self.parameters["MINchl2nBG"] = 0.50 * self.redf[10] * self.redf[15]
        self.parameters["MAXchl2nBG"] = 3.83 * self.redf[10] * self.redf[15]
        self.parameters["alfaPs"] = 0.0393 * self.redf[0] * self.redf[5]
        self.parameters["alfaPl"] = 0.0531 * self.redf[0] * self.redf[5]
        self.parameters["alfaBG"] = 0.0393 * self.redf[0] * self.redf[5]

        if self.community_export:
            # community sinking constants
            self.parameters["sinkDiaD"] = torch.tensor(5.0 / 86400, device=device, dtype=dtype)
            self.parameters["sinkFlaD"] = torch.tensor(5.0 / 86400, device=device, dtype=dtype)
            self.parameters["sinkMicD"] = torch.tensor(5.0 / 86400, device=device, dtype=dtype)
            self.parameters["sinkMesD"] = torch.tensor(5.0 / 86400, device=device, dtype=dtype)
            self.parameters["sinkBgD"] = torch.tensor(5.0 / 86400, device=device, dtype=dtype)
            self.parameters["sinkFish1D"] = torch.tensor(5.0 / 86400, device=device, dtype=dtype)
            self.parameters["sinkFish2D"] = torch.tensor(5.0 / 86400, device=device, dtype=dtype)

        self.tracer_names.extend(
            [
                f"{self.model_name}_no3",
                f"{self.model_name}_nh4",
                f"{self.model_name}_pho",
                f"{self.model_name}_sil",
                f"{self.model_name}_oxy",
                f"{self.model_name}_fla",
                f"{self.model_name}_dia",
            ]
        )
        if self.use_cyanos:
            self.tracer_names.append(f"{self.model_name}_bg")
            if self.use_chl:
                self.tracer_names.append(f"{self.model_name}_bgchl")
        if self.use_chl:
            self.tracer_names.extend([f"{self.model_name}_diachl", f"{self.model_name}_flachl"])
        self.tracer_names.extend(
            [
                f"{self.model_name}_microzoo",
                f"{self.model_name}_mesozoo",
                f"{self.model_name}_det",
                f"{self.model_name}_opa",
                f"{self.model_name}_dom",
            ]
        )
        self.benthic_tracer_names.extend(
            [
                f"{self.model_name}_sed1",
                f"{self.model_name}_sed2",
                f"{self.model_name}_sed3",
            ]
        )
        if self.use_fish:
            self.tracer_names.extend(
                [
                    f"{self.model_name}_fish1",
                    f"{self.model_name}_fish2",
                ]
            )
            self.benthic_tracer_names.append(f"{self.model_name}_mb1")

        self.dependencies.extend(["temp", "salt", "taub"])
        if self.use_fish:
            self.dependencies.append("h")

        self.setup_internal_fields()

    def setup_internal_fields(self):
        """Routine for setting up internal fields that depend on internal parameters."""
        device = self.redf.device
        dtype = self.redf.dtype

        _zero = torch.zeros_like(self.parameters["sinkDet"], requires_grad=False)
        _ws = [_zero, _zero, _zero, _zero, _zero, _zero, -self.parameters["sinkDia"]]
        if self.use_cyanos:
            _ws.append(-self.parameters["sinkBG"])
            if self.use_chl:
                _ws.append(-self.parameters["sinkBG"])
        if self.use_chl:
            _ws.append(-self.parameters["sinkDia"])
            _ws.append(_zero)
        _ws.extend([_zero, _zero, -self.parameters["sinkDet"], -self.parameters["sinkOPAL"], _zero])
        if self.use_fish:
            _ws.extend([_zero, _zero])
        self.ws = torch.stack(_ws).unsqueeze(0).unsqueeze(-1)

        _minimum = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0e-7, 1.0e-7]
        if self.use_cyanos:
            _minimum.append(1.0e-14)
            if self.use_chl:
                _minimum.append(1.0e-14 / 20.0)
        if self.use_chl:
            _minimum.append(1.0e-7 / 27.0)
            _minimum.append(1.0e-7 / 20.0)
        _minimum.extend([1.0e-7, 1.0e-7, 0.0, 0.0, 0.0])
        if self.use_fish:
            _minimum.append(0.0)
            _minimum.append(0.0)
        self.minimum = torch.tensor(_minimum, device=device, dtype=dtype).unsqueeze(0).unsqueeze(-1)

        n_benthic = 4 if self.use_fish else 3
        self.minimum_td = TensorDict(
            {"pelagic": self.minimum, "benthic": torch.zeros((1, n_benthic), dtype=dtype, device=device)}
        )

        if self.use_cyanos:
            if self.use_chl:
                # use diachl, flachl, bgchl, det, dom
                self.light_attenuation_indexes = [9, 10, 8, 13, 15]
            else:
                # use fla, dia, bg, det, dom
                self.light_attenuation_indexes = [5, 6, 7, 10, 12]
            light_scale_list = [
                self.parameters["Exphy"],
                self.parameters["Exphy"],
                self.parameters["Exphy"],
                self.parameters["extdet"],
                self.parameters["extdom"],
            ]
        else:
            if self.use_chl:
                # use diachl, flachl, det, dom
                self.light_attenuation_indexes = [7, 8, 11, 13]
            else:
                # use fla, dia, det, dom
                self.light_attenuation_indexes = [5, 6, 9, 11]
            light_scale_list = [
                self.parameters["Exphy"],
                self.parameters["Exphy"],
                self.parameters["extdet"],
                self.parameters["extdom"],
            ]
        self.light_attenuation_scale_factor = torch.stack(light_scale_list).unsqueeze(-1)
        self.light_attenuation_background = self.parameters["EXw"]

    @classmethod
    def create_model_from_yaml_file(
        cls,
        path_fabm: str,
        path_gotm: str,
        use_corrected_bioc: bool = False,
        device: str = "cpu",
        dtype: Union[str, torch.dtype] = torch.float64,
        *args,
        **kwargs,
    ):
        """Creates a NerscEcosmoOperationalModel instance from .yaml configuration files. Should generally be used
        instead of the constructor. Designed to be compatible with the configuration files of GOTM.

        Args:
            path_fabm (str): Path to the FABM .yaml configuration file
            path_gotm (str): Path to the GOTM .yaml configuration file
            use_corrected_bioc (bool, optional): Whether to use the "corrected" scaling of the Exphy parameter
                depending on *use_chl*. Defaults to False.
            device (str, optional): Torch device. Defaults to "cpu".
            dtype (Union[str, torch.dtype], optional): Torch data type. Defaults to torch.float64.

        Returns:
            NerscEcosmoOperationalModel: The generated model
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
            if config_instance["model"].replace("_", "/") == "nersc/ecosmo":
                if model is not None:
                    raise ValueError(f"More than one Ecosmo operational model detected in file {file_fabm}!")

                try:
                    params = [
                        "use_cyanos",
                        "use_fish",
                        "couple_co2",
                        "use_chl",
                        "community_export",
                    ]
                    bools = [config_instance["parameters"][param] for param in params]
                except (KeyError, ValueError, TypeError):
                    raise ValueError(
                        f"instances/{name_instance}/parameters/[{params}] missing or invalid "
                        f"in {path_fabm}. Expected type bool."
                    )
                model = cls(
                    *bools,
                    use_corrected_bioc=use_corrected_bioc,
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
        model.parameters["nfixation_minimum_daily_par"] = torch.tensor(
            config_par["nfixation_minimum_daily_par"], device=device, dtype=dtype
        )
        model.parameters["bg_growth_minimum_daily_rad"] = torch.tensor(
            config_par["bg_growth_minimum_daily_rad"], device=device, dtype=dtype
        )
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
        model.parameters["surface_deposition_sil"] = (
            config_par["surface_deposition_sil"] * model.redf[2] * model.redf[5] / 86400
        )

        # parameters stored in the array bio_c in fortran
        model.parameters["muPl"] = torch.tensor(config_par["muPl"] / 86400, device=device, dtype=dtype)
        model.parameters["muPs"] = torch.tensor(config_par["muPs"] / 86400, device=device, dtype=dtype)
        model.parameters["aa"] = torch.tensor(config_par["aa"], device=device, dtype=dtype)
        model.parameters["EXw"] = torch.tensor(config_par["EXw"], device=device, dtype=dtype)
        model.parameters["Exphy"] = (
            torch.tensor(config_par["Exphy"], device=device, dtype=dtype)
            if (model.use_chl and use_corrected_bioc)
            else (config_par["Exphy"] / (model.redf[0] * model.redf[5]))
        )
        model.parameters["rNH4"] = config_par["rNH4"] * model.redf[0] * model.redf[5]
        model.parameters["rNO3"] = config_par["rNO3"] * model.redf[0] * model.redf[5]
        model.parameters["psi"] = config_par["psi"] / (model.redf[0] * model.redf[5])
        model.parameters["mPl"] = torch.tensor(config_par["mPl"] / 86400, device=device, dtype=dtype)
        model.parameters["mPs"] = torch.tensor(config_par["mPs"] / 86400, device=device, dtype=dtype)
        model.parameters["GrZlP"] = torch.tensor(config_par["GrZlP"] / 86400, device=device, dtype=dtype)
        model.parameters["GrZsP"] = torch.tensor(config_par["GrZsP"] / 86400, device=device, dtype=dtype)
        model.parameters["GrZlZ"] = torch.tensor(config_par["GrZlZ"] / 86400, device=device, dtype=dtype)
        model.parameters["Rg"] = config_par["Rg"] * model.redf[0] * model.redf[5]
        model.parameters["mZl"] = torch.tensor(config_par["mZl"] / 86400, device=device, dtype=dtype)
        model.parameters["mZs"] = torch.tensor(config_par["mZs"] / 86400, device=device, dtype=dtype)
        model.parameters["excZl"] = torch.tensor(config_par["excZl"] / 86400, device=device, dtype=dtype)
        model.parameters["excZs"] = torch.tensor(config_par["excZs"] / 86400, device=device, dtype=dtype)
        model.parameters["gammaZlp"] = torch.tensor(config_par["gammaZlp"], device=device, dtype=dtype)
        model.parameters["gammaZsp"] = torch.tensor(config_par["gammaZsp"], device=device, dtype=dtype)
        model.parameters["gammaZd"] = torch.tensor(config_par["gammaZd"], device=device, dtype=dtype)
        model.parameters["reminD"] = torch.tensor(config_par["reminD"] / 86400, device=device, dtype=dtype)
        model.parameters["sinkDet"] = torch.tensor(config_par["sinkDet"] / 86400, device=device, dtype=dtype)
        model.parameters["Wa"] = torch.tensor(config_par["Wa"] / 86400, device=device, dtype=dtype)
        model.parameters["rPO4"] = config_par["rPO4"] * model.redf[1] * model.redf[5]
        model.parameters["rSi"] = config_par["rSi"] * model.redf[2] * model.redf[5]
        model.parameters["regenSi"] = torch.tensor(config_par["regenSi"] / 86400, device=device, dtype=dtype)
        model.parameters["muBG"] = torch.tensor(config_par["muBG"] / 86400, device=device, dtype=dtype)
        model.parameters["TctrlBG"] = torch.tensor(config_par["TctrlBG"], device=device, dtype=dtype)
        model.parameters["TrefBG"] = torch.tensor(config_par["TrefBG"], device=device, dtype=dtype)
        model.parameters["GrBG"] = torch.tensor(config_par["GrBG"] / 86400, device=device, dtype=dtype)
        model.parameters["mBG"] = torch.tensor(config_par["mBG"] / 86400, device=device, dtype=dtype)
        model.parameters["upliftBG"] = torch.tensor(config_par["upliftBG"] / 86400, device=device, dtype=dtype)
        model.parameters["crBotStr"] = torch.tensor(config_par["crBotStr"], device=device, dtype=dtype)
        model.parameters["resuspRt"] = torch.tensor(
            config_par["resuspRt"] / 86400, device=device, dtype=dtype
        )  # resuspRt
        model.parameters["sedimRt"] = torch.tensor(config_par["sedimRt"] / 86400, device=device, dtype=dtype)  # sedimRt
        model.parameters["burialRt"] = torch.tensor(
            config_par["burialRt"] / 86400, device=device, dtype=dtype
        )  # burialRt
        model.parameters["reminSED"] = torch.tensor(
            config_par["reminSED"] / 86400, device=device, dtype=dtype
        )  # reminSED
        model.parameters["TctrlDenit"] = torch.tensor(
            config_par["TctrlDenit"], device=device, dtype=dtype
        )  # TctrlDenit
        model.parameters["RelSEDp1"] = torch.tensor(config_par["RelSEDp1"], device=device, dtype=dtype)  # RelSEDp1
        model.parameters["RelSEDp2"] = torch.tensor(config_par["RelSEDp2"], device=device, dtype=dtype)  # RelSEDp2
        model.parameters["reminSEDsi"] = torch.tensor(
            config_par["reminSEDsi"] / 86400, device=device, dtype=dtype
        )  # reminSEDsi
        model.parameters["sinkOPAL"] = torch.tensor(
            config_par["sinkOPAL"] / 86400, device=device, dtype=dtype
        )  # sinkOPAL
        model.parameters["sinkBG"] = torch.tensor(config_par["sinkBG"] / 86400, device=device, dtype=dtype)  # sinkBG
        model.parameters["sinkDia"] = torch.tensor(config_par["sinkDia"] / 86400, device=device, dtype=dtype)  # sinkDia

        model.parameters["extdet"] = torch.tensor(config_par["Exdet"] / 1000, device=device, dtype=dtype)
        model.parameters["extdom"] = torch.tensor(config_par["Exdom"] / 1000, device=device, dtype=dtype)

        #  growth fractions
        model.parameters["prefZsPs"] = torch.tensor(config_par["prefZsPs"], device=device, dtype=dtype)
        model.parameters["prefZsPl"] = torch.tensor(config_par["prefZsPl"], device=device, dtype=dtype)
        model.parameters["prefZsD"] = torch.tensor(config_par["prefZsD"], device=device, dtype=dtype)
        model.parameters["prefZsBG"] = torch.tensor(config_par["prefZsBG"], device=device, dtype=dtype)
        model.parameters["prefZlPs"] = torch.tensor(config_par["prefZlPs"], device=device, dtype=dtype)
        model.parameters["prefZlPl"] = torch.tensor(config_par["prefZlPl"], device=device, dtype=dtype)
        model.parameters["prefZlZs"] = torch.tensor(config_par["prefZlZs"], device=device, dtype=dtype)
        model.parameters["prefZlD"] = torch.tensor(config_par["prefZlD"], device=device, dtype=dtype)
        model.parameters["prefZlBG"] = torch.tensor(config_par["prefZlBG"], device=device, dtype=dtype)

        if model.use_fish:
            # Macrobenthos parameter settings by Ute
            model.parameters["mMB1"] = torch.tensor(config_par["mMB1"] / 86400, device=device, dtype=dtype)
            model.parameters["excMB1"] = torch.tensor(config_par["excMB1"] / 86400, device=device, dtype=dtype)
            model.parameters["rMB1"] = config_par["rMB1"] * model.redf[0] * model.redf[5]
            model.parameters["GrMB1Z"] = torch.tensor(config_par["GrMB1Z"] / 86400, device=device, dtype=dtype)
            model.parameters["GrMB1P"] = torch.tensor(config_par["GrMB1P"] / 86400, device=device, dtype=dtype)
            model.parameters["GrMB1Det"] = torch.tensor(config_par["GrMB1Det"] / 86400, device=device, dtype=dtype)
            model.parameters["GrMB1Sed"] = torch.tensor(config_par["GrMB1Sed"] / 86400, device=device, dtype=dtype)
            model.parameters["tempcMB1"] = torch.tensor(config_par["tempcMB1"], device=device, dtype=dtype)
            model.parameters["prefMB1Zs"] = torch.tensor(config_par["prefMB1Zs"], device=device, dtype=dtype)
            model.parameters["prefMB1Zl"] = torch.tensor(config_par["prefMB1Zl"], device=device, dtype=dtype)
            model.parameters["prefMB1Det"] = torch.tensor(config_par["prefMB1Det"], device=device, dtype=dtype)
            model.parameters["prefMB1Sed"] = torch.tensor(config_par["prefMB1Sed"], device=device, dtype=dtype)
            model.parameters["prefMB1P"] = torch.tensor(config_par["prefMB1P"], device=device, dtype=dtype)
            model.parameters["prefMB1Dom"] = torch.tensor(config_par["prefMB1Dom"], device=device, dtype=dtype)

            # parameter for fish1 functional group
            model.parameters["mF1"] = torch.tensor(config_par["mF1"] / 86400, device=device, dtype=dtype)
            model.parameters["excF1"] = torch.tensor(config_par["excF1"] / 86400, device=device, dtype=dtype)
            model.parameters["rF1"] = config_par["rF1"] * model.redf[0] * model.redf[5]
            model.parameters["rF1MB"] = config_par["rF1MB"] * model.redf[0] * model.redf[5]
            model.parameters["tempcF1"] = torch.tensor(config_par["tempcF1"], device=device, dtype=dtype)
            model.parameters["asefF1"] = torch.tensor(config_par["asefF1"], device=device, dtype=dtype)
            model.parameters["GrF1Zl"] = torch.tensor(config_par["GrF1Zl"] / 86400, device=device, dtype=dtype)
            model.parameters["GrF1Zs"] = torch.tensor(config_par["GrF1Zs"] / 86400, device=device, dtype=dtype)
            model.parameters["GrF1Det"] = torch.tensor(config_par["GrF1Det"] / 86400, device=device, dtype=dtype)
            model.parameters["GrF1MB1"] = torch.tensor(config_par["GrF1MB1"] / 86400, device=device, dtype=dtype)
            model.parameters["prefF1Zl"] = torch.tensor(config_par["prefF1Zl"], device=device, dtype=dtype)
            model.parameters["prefF1Zs"] = torch.tensor(config_par["prefF1Zs"], device=device, dtype=dtype)
            model.parameters["prefF1Det"] = torch.tensor(config_par["prefF1Det"], device=device, dtype=dtype)
            model.parameters["prefF1MB1"] = torch.tensor(config_par["prefF1MB1"], device=device, dtype=dtype)
            # parameter for fish2 functional group
            model.parameters["mF2"] = torch.tensor(config_par["mF2"] / 86400, device=device, dtype=dtype)
            model.parameters["excF2"] = torch.tensor(config_par["excF2"] / 86400, device=device, dtype=dtype)
            model.parameters["rF2"] = config_par["rF2"] * model.redf[0] * model.redf[5]
            model.parameters["rF2MB"] = config_par["rF2MB"] * model.redf[0] * model.redf[5]
            model.parameters["tempcF2"] = torch.tensor(config_par["tempcF2"], device=device, dtype=dtype)
            model.parameters["asefF2"] = torch.tensor(config_par["asefF2"], device=device, dtype=dtype)
            model.parameters["GrF2Zl"] = torch.tensor(config_par["GrF2Zl"] / 86400, device=device, dtype=dtype)
            model.parameters["GrF2F1"] = torch.tensor(config_par["GrF2F1"] / 86400, device=device, dtype=dtype)
            model.parameters["GrF2Det"] = torch.tensor(config_par["GrF2Det"] / 86400, device=device, dtype=dtype)
            model.parameters["GrF2MB1"] = torch.tensor(config_par["GrF2MB1"] / 86400, device=device, dtype=dtype)
            model.parameters["prefF2Zl"] = torch.tensor(config_par["prefF2Zl"], device=device, dtype=dtype)
            model.parameters["prefF2F1"] = torch.tensor(config_par["prefF2F1"], device=device, dtype=dtype)
            model.parameters["prefF2Det"] = torch.tensor(config_par["prefF2Det"], device=device, dtype=dtype)
            model.parameters["prefF2MB1"] = torch.tensor(config_par["prefF2MB1"], device=device, dtype=dtype)

        # chlorophyll-a constants
        model.parameters["MINchl2nPs"] = config_par["MINchl2nPs"] * model.redf[10] * model.redf[15]
        model.parameters["MAXchl2nPs"] = config_par["MAXchl2nPs"] * model.redf[10] * model.redf[15]
        model.parameters["MINchl2nPl"] = config_par["MINchl2nPl"] * model.redf[10] * model.redf[15]
        model.parameters["MAXchl2nPl"] = config_par["MAXchl2nPl"] * model.redf[10] * model.redf[15]
        model.parameters["MINchl2nBG"] = config_par["MINchl2nBG"] * model.redf[10] * model.redf[15]
        model.parameters["MAXchl2nBG"] = config_par["MAXchl2nBG"] * model.redf[10] * model.redf[15]
        model.parameters["alfaPs"] = config_par["alfaPs"] * model.redf[0] * model.redf[5]
        model.parameters["alfaPl"] = config_par["alfaPl"] * model.redf[0] * model.redf[5]
        model.parameters["alfaBG"] = config_par["alfaBG"] * model.redf[0] * model.redf[5]

        if model.community_export:
            # community sinking constants
            model.parameters["sinkDiaD"] = torch.tensor(config_par["sinkDiaD"] / 86400, device=device, dtype=dtype)
            model.parameters["sinkFlaD"] = torch.tensor(config_par["sinkFlaD"] / 86400, device=device, dtype=dtype)
            model.parameters["sinkMicD"] = torch.tensor(config_par["sinkMicD"] / 86400, device=device, dtype=dtype)
            model.parameters["sinkMesD"] = torch.tensor(config_par["sinkMesD"] / 86400, device=device, dtype=dtype)
            model.parameters["sinkBgD"] = torch.tensor(config_par["sinkBgD"] / 86400, device=device, dtype=dtype)
            model.parameters["sinkFish1D"] = torch.tensor(config_par["sinkFish1D"] / 86400, device=device, dtype=dtype)
            model.parameters["sinkFish2D"] = torch.tensor(config_par["sinkFish2D"] / 86400, device=device, dtype=dtype)

        model.setup_internal_fields()

        return model

    def update_parameters(self, parameters):
        """Update the parameters of the model.

        Args:
            parameters (TensorDict): The model parameters to be updated
        """
        super().update_parameters(parameters)

        if any(k in parameters for k in ("sinkDia", "sinkBG", "sinkDet", "sinkOPAL")):
            _zero = torch.zeros_like(self.parameters["sinkDet"], requires_grad=False)
            _ws = [_zero, _zero, _zero, _zero, _zero, _zero, -self.parameters["sinkDia"]]
            if self.use_cyanos:
                _ws.append(-self.parameters["sinkBG"])
                if self.use_chl:
                    _ws.append(-self.parameters["sinkBG"])
            if self.use_chl:
                _ws.append(-self.parameters["sinkDia"])
                _ws.append(_zero)
            _ws.extend([_zero, _zero, -self.parameters["sinkDet"], -self.parameters["sinkOPAL"], _zero])
            self.ws = torch.stack(_ws).unsqueeze(0).unsqueeze(-1)

        if any(k in parameters for k in ("Exphy", "extdet", "extdom")):
            if self.use_cyanos:
                light_scale_list = [
                    self.parameters["Exphy"],
                    self.parameters["Exphy"],
                    self.parameters["Exphy"],
                    self.parameters["extdet"],
                    self.parameters["extdom"],
                ]
            else:
                light_scale_list = [
                    self.parameters["Exphy"],
                    self.parameters["Exphy"],
                    self.parameters["extdet"],
                    self.parameters["extdom"],
                ]
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
                If this model is not coupled to a light model, should also have the corresponding light fields
                "par", "mean_par", "mean_surface_par".

        Returns:
            TensorDict: Vector of temporal derivatives
        """

        use_state_light = use_state_par_mean = use_state_par0_mean = False

        if isinstance(concentration, torch.Tensor):
            x = concentration
        elif isinstance(concentration, TensorDict):
            x = concentration["pelagic"]
            use_state_light = "light_par" in concentration
            if self.use_cyanos:
                use_state_par_mean = "light_par_mean" in concentration
                use_state_par0_mean = "light_par0_mean" in concentration

        no3 = x[..., 0, :]
        nh4 = x[..., 1, :]
        pho = x[..., 2, :]
        sil = x[..., 3, :]
        oxy = x[..., 4, :]
        fla = x[..., 5, :]
        dia = x[..., 6, :]
        i = 0
        if self.use_cyanos:
            bg = x[..., 7, :]
            i += 1
            if self.use_chl:
                bgchl = x[..., 8, :]
                i += 1
        else:
            bg = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        if self.use_chl:
            diachl = x[..., 7 + i, :]
            flachl = x[..., 8 + i, :]
            i += 2
        microzoo = x[..., 7 + i, :]
        mesozoo = x[..., 8 + i, :]
        det = x[..., 9 + i, :]
        opa = x[..., 10 + i, :]
        dom = x[..., 11 + i, :]
        if self.use_fish:
            raise NotImplementedError("fish are currently not implemented")
            # fish1 = x[..., 12 + i, :]
            # fish2 = x[..., 13 + i, :]

        if use_state_light:
            par = concentration["light_par"]
        else:
            par = dynamic_inputs["par"]

        if self.use_cyanos:
            if use_state_par_mean:
                mean_par = concentration["light_par_mean"]
            else:
                mean_par = dynamic_inputs["mean_par"]
            if use_state_par0_mean:
                mean_surface_par = concentration["light_par0_mean"]
            else:
                mean_surface_par = dynamic_inputs["mean_surface_par"]

            if mean_surface_par.ndim < mean_par.ndim:
                mean_surface_par = mean_surface_par.unsqueeze(-1)

        temp = dynamic_inputs["temp"]
        salt = dynamic_inputs["salt"]

        if self.use_fish:
            raise NotImplementedError("fish are currently not implemented")
            # h = dynamic_inputs["h"]
            # if h.ndim > mesozoo.ndim:
            #     h = h[..., 0, :]
            # mesoi = (mesozoo * h).sum(axis=-1)
            # microi = (microzoo * h).sum(axis=-1)
            # deti = (det * h).sum(axis=-1)
            # fishi1 = (fish1 * h).sum(axis=-1)
            # fishi2 = (fish2 * h).sum(axis=-1)

        zero = torch.tensor(0.0).to(fla)
        one = torch.tensor(1.0).to(fla)

        fla_loss = self.mask_func(0.01 - fla, zero, one)
        dia_loss = self.mask_func(0.01 - dia, zero, one)
        bg_loss = self.mask_func(0.01 - bg, zero, one)
        mic_loss = self.mask_func(0.001 - microzoo, zero, one)
        mes_loss = self.mask_func(0.001 - mesozoo, zero, one)

        # t dependency for fish and MB
        if self.use_fish:
            temp / ((temp + 273.15) * 273.15)

        # remineralisation rate
        frem = self.parameters["reminD"] * (1.0 + 20.0 * (temp**2 / (13.0**2 + temp**2)))
        fremDOM = 10.0 * frem

        # nutrient limitation factors
        # k denotes half-saturation values
        up_nh4 = nh4 / (self.parameters["rNH4"] + nh4)
        up_no3 = no3 / (self.parameters["rNO3"] + no3) * torch.exp(-self.parameters["psi"] * nh4)
        up_n = up_nh4 + up_no3
        up_pho = pho / (self.parameters["rPO4"] + pho)
        t_sil = torch.clamp_min(sil - 80.0, 0.0)
        up_sil = t_sil / (self.parameters["rSi"] + t_sil)

        # temperature dependence
        Ts = 1.0
        Tl = 1.0
        if self.use_cyanos:
            Tbg = self.mask_func(
                mean_surface_par - self.parameters["bg_growth_minimum_daily_rad"],
                self.mask_func(
                    salt - 10.0,
                    0.0,
                    1.0 / (1.0 + torch.exp(self.parameters["TctrlBG"] * (self.parameters["TrefBG"] - temp))),
                ),
                0.0,
            )

        # production and nutrient uptake
        blight = torch.clamp_min(torch.tanh(self.parameters["aa"] * par), 0.0)
        Ps_prod = Ts * torch.minimum(blight, torch.minimum(up_n, up_pho))
        Pl_prod = Tl * torch.minimum(blight, torch.minimum(up_n, torch.minimum(up_pho, up_sil)))

        Prod = self.parameters["muPl"] * Pl_prod * dia + self.parameters["muPs"] * Ps_prod * fla

        if self.use_cyanos:
            Bg_prod = Tbg * torch.minimum(blight, torch.minimum(up_n, up_pho))
            Prod = Prod + self.parameters["muBG"] * Bg_prod * bg  # cyanobacteria production

            Bg_fix = self.mask_func(
                mean_par - self.parameters["nfixation_minimum_daily_par"],
                Tbg * torch.minimum(blight, up_pho) - Bg_prod,
                0.0,
            )
        else:
            Bg_fix = 0.0

        if self.use_chl:
            num_fla = (
                self.parameters["MAXchl2nPs"] * torch.clamp_min(Ps_prod, 0.1) * self.parameters["muPs"] * 86400 * fla
            )
            den_fla = self.parameters["alfaPs"] * par * flachl
            num_fla = torch.where(
                den_fla > 0, num_fla, 10 * self.parameters["MAXchl2nPs"]
            )  # prevent nans in backward pass
            den_fla = torch.where(den_fla > 0, den_fla, 1.0)
            chl2c_fla = num_fla / den_fla
            num_dia = (
                self.parameters["MAXchl2nPl"] * torch.clamp_min(Pl_prod, 0.1) * self.parameters["muPl"] * 86400 * dia
            )
            den_dia = self.parameters["alfaPl"] * par * diachl
            num_dia = torch.where(
                den_dia > 0, num_dia, 10 * self.parameters["MAXchl2nPl"]
            )  # prevent nans in backward pass
            den_dia = torch.where(den_dia > 0, den_dia, 1.0)
            chl2c_dia = num_dia / den_dia

            chl2c_fla = torch.clamp_min(chl2c_fla, self.parameters["MINchl2nPs"])
            chl2c_fla = torch.clamp_max(chl2c_fla, self.parameters["MAXchl2nPs"])
            chl2c_dia = torch.clamp_min(chl2c_dia, self.parameters["MINchl2nPl"])
            chl2c_dia = torch.clamp_max(chl2c_dia, self.parameters["MAXchl2nPl"])

            if self.use_cyanos:
                num_bg = (
                    self.parameters["MAXchl2nBG"] * torch.clamp_min(Bg_prod, 0.1) * self.parameters["muBG"] * 86400 * bg
                )
                den_bg = self.parameters["alfaBG"] * par * bgchl
                num_bg = torch.where(
                    den_bg > 0, num_bg, 10 * self.parameters["MAXchl2nBG"]
                )  # prevent nans in backward pass
                den_bg = torch.where(den_bg > 0, den_bg, 1.0)
                chl2c_bg = num_bg / den_bg
                chl2c_bg = torch.maximum(self.parameters["MINchl2nBG"], chl2c_bg)
                chl2c_bg = torch.minimum(self.parameters["MAXchl2nBG"], chl2c_bg)

        Fs = (
            self.parameters["prefZsPs"] * fla
            + self.parameters["prefZsPl"] * dia
            + self.parameters["prefZsD"] * det
            # + self.parameters["prefZsBG"] * bg
        )
        Fl = (
            self.parameters["prefZlPs"] * fla
            + self.parameters["prefZlPl"] * dia
            + self.parameters["prefZlZs"] * microzoo
            + self.parameters["prefZlD"] * det
            # + self.parameters["prefZlBG"] * bg
        )
        if self.use_cyanos:
            Fs = Fs + self.parameters["prefZsBG"] * bg
            Fl = Fl + self.parameters["prefZlBG"] * bg

        ZsonPs = fla_loss * self.parameters["GrZsP"] * self.parameters["prefZsPs"] * fla / (self.parameters["Rg"] + Fs)
        ZsonPl = dia_loss * self.parameters["GrZsP"] * self.parameters["prefZsPl"] * dia / (self.parameters["Rg"] + Fs)
        ZsonD = self.parameters["GrZsP"] * self.parameters["prefZsD"] * det / (self.parameters["Rg"] + Fs)

        ZlonPs = fla_loss * self.parameters["GrZlP"] * self.parameters["prefZlPs"] * fla / (self.parameters["Rg"] + Fl)
        ZlonPl = dia_loss * self.parameters["GrZlP"] * self.parameters["prefZlPl"] * dia / (self.parameters["Rg"] + Fl)
        ZlonD = self.parameters["GrZlP"] * self.parameters["prefZlD"] * det / (self.parameters["Rg"] + Fl)
        ZlonZs = (
            mic_loss * self.parameters["GrZlZ"] * self.parameters["prefZlZs"] * microzoo / (self.parameters["Rg"] + Fl)
        )

        if self.use_cyanos:
            ZsonBg = bg_loss * self.parameters["GrBG"] * self.parameters["prefZsBG"] * bg / (self.parameters["Rg"] + Fs)
            ZlonBg = bg_loss * self.parameters["GrBG"] * self.parameters["prefZlBG"] * bg / (self.parameters["Rg"] + Fl)
        else:
            ZsonBg = 0.0
            ZlonBg = 0.0

        # nitrification
        Onitr = 0.01 * self.redf[6]  # according to Neumann  (Onitr in mlO2/l see also Stigebrand and Wulff)
        bioom1 = self.mask_func(oxy, 0.1 / 86400 * torch.exp(temp * 0.11) * oxy / (Onitr + oxy), 0.0)
        bioom3 = 0.0
        bioom5 = self.mask_func(no3, self.mask_func(oxy, 0.0, 5.0), 0.0)
        bioom6 = self.mask_func(oxy, 1.0, 0.0)
        bioom7 = self.mask_func(no3, 0.0, self.mask_func(oxy, 0.0, 1.0))
        # bioom8 = self.mask_func(no3, self.mask_func(oxy, 0.0, 1.0), 0.0)

        d_fla = (
            (self.parameters["muPs"] * Ps_prod - self.parameters["mPs"] * fla_loss) * fla
            - ZsonPs * microzoo
            - ZlonPs * mesozoo
        )
        d_dia = (
            (self.parameters["muPl"] * Pl_prod - self.parameters["mPl"] * dia_loss) * dia
            - ZsonPl * microzoo
            - ZlonPl * mesozoo
        )
        if self.use_cyanos:
            d_bg = (
                (self.parameters["muBG"] * (Bg_prod + Bg_fix) - self.parameters["mBG"] * bg_loss) * bg
                - ZsonBg * microzoo
                - ZlonBg * mesozoo
            )

        # for chlorophyll-a
        if self.use_chl:
            d_flachl = self.parameters["muPs"] * Ps_prod * chl2c_fla * fla - (
                (self.parameters["mPs"] * fla_loss * fla + ZsonPs * microzoo + ZlonPs * mesozoo) * flachl / fla
            )
            d_diachl = self.parameters["muPl"] * Pl_prod * chl2c_dia * dia - (
                (self.parameters["mPl"] * dia * dia_loss + ZsonPl * microzoo + ZlonPl * mesozoo) * diachl / dia
            )
            if self.use_cyanos:
                d_bgchl = self.parameters["muBG"] * (Bg_prod + Bg_fix) * chl2c_bg * bg - (
                    (self.parameters["mBG"] * bg * bg_loss + ZsonBg * microzoo + ZlonBg * mesozoo) * bgchl / bg
                )

        # microzooplankton
        Zs_prod = self.parameters["gammaZsp"] * (ZsonPs + ZsonPl + ZsonBg) + self.parameters["gammaZd"] * ZsonD
        d_microzoo = (
            Zs_prod - (self.parameters["mZs"] + self.parameters["excZs"] + self.parameters["zpr"]) * mic_loss
        ) * microzoo - ZlonZs * mesozoo

        # mesozooplankton
        Zl_prod = self.parameters["gammaZlp"] * (ZlonPs + ZlonPl + ZlonBg + ZlonZs) + self.parameters["gammaZd"] * ZlonD
        d_mesozoo = (
            Zl_prod - (self.parameters["mZl"] + self.parameters["excZl"] + self.parameters["zpr"]) * mes_loss
        ) * mesozoo

        # detritus
        dxxdet = (
            (
                (1.0 - self.parameters["gammaZsp"]) * (ZsonPs + ZsonPl + ZsonBg)
                + (1.0 - self.parameters["gammaZd"]) * ZsonD
            )
            * microzoo
            + (
                (1.0 - self.parameters["gammaZlp"]) * (ZlonPs + ZlonPl + ZlonBg + ZlonZs)
                + (1.0 - self.parameters["gammaZd"]) * ZlonD
            )
            * mesozoo
            + self.parameters["mZs"] * microzoo * mic_loss
            + self.parameters["mZl"] * mesozoo * mes_loss
            + self.parameters["mPs"] * fla * fla_loss
            + self.parameters["mPl"] * dia * dia_loss
        )
        if self.use_cyanos:
            dxxdet = dxxdet + (self.parameters["mBG"] * bg * bg_loss)

        d_det = (1.0 - self.parameters["frr"]) * dxxdet - ZsonD * microzoo - ZlonD * mesozoo - frem * det

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
            + self.parameters["excZs"] * microzoo * mic_loss
            + self.parameters["excZl"] * mesozoo * mes_loss
            + frem * det
            + fremDOM * dom
            - bioom1 * nh4
        )
        d_nh4 = rhs_amm

        # phosphate
        rhs_pho = (
            -Prod
            - self.parameters["muBG"] * bg * Bg_fix
            + self.parameters["excZs"] * microzoo * mic_loss
            + self.parameters["excZl"] * mesozoo * mes_loss
            + frem * det
            + fremDOM * dom
        )
        d_pho = rhs_pho

        # silicate
        d_sil = -self.parameters["muPl"] * Pl_prod * dia + self.parameters["regenSi"] * opa

        # opal
        d_opa = (
            self.parameters["mPl"] * dia * dia_loss
            + ZsonPl * microzoo
            + ZlonPl * mesozoo
            - self.parameters["regenSi"] * opa
        )

        # oxygen
        rhs_oxy = (
            (
                (6.625 * up_nh4 + 8.125 * up_no3 + 1.0e-10) / (up_n + 1.0e-10) * Prod
                - bioom6
                * 6.625
                * (self.parameters["excZs"] * microzoo * mic_loss + self.parameters["excZl"] * mesozoo * mes_loss)
                - frem * det * (bioom6 + bioom7) * 6.625
                - (bioom6 + bioom7) * 6.625 * fremDOM * dom
                - 2.0 * bioom1 * nh4
            )
            * self.redf[10]
            * self.redf[15]
        )
        d_oxy = rhs_oxy

        # Carbonate dynamics
        # To be used by future extension
        # if self.couple_co2:
        #     d_dic = self.redf[15] * (
        #         self.parameters["excZs"] * microzoo * mic_loss
        #         + self.parameters["excZl"] * mesozoo * mes_loss
        #         + frem * det
        #         + fremDOM * dom
        #         - Prod
        #     )

        #     self.redf[15] * (rhs_amm - rhs_nit) * self.redf[10] - 0.5 * rhs_oxy * (1.0 - bioom6)

        dx_list = [d_no3, d_nh4, d_pho, d_sil, d_oxy, d_fla, d_dia]
        if self.use_cyanos:
            dx_list.append(d_bg)
            if self.use_chl:
                dx_list.append(d_bgchl)
        if self.use_chl:
            dx_list.extend([d_diachl, d_flachl])
        dx_list.extend([d_microzoo, d_mesozoo, d_det, d_opa, d_dom])
        dx = torch.stack(dx_list, axis=-2)

        if isinstance(concentration, TensorDict):
            return TensorDict({"pelagic": dx}, batch_size=concentration.batch_size)
        else:
            return dx

    def do_bottom(self, concentration: TensorDict, dynamic_inputs, *args, **kwargs) -> TensorDict:
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
        oxy_bottom = x[..., 4, 0]
        i = 0
        if self.use_cyanos:
            i += 1
            if self.use_chl:
                i += 1
        if self.use_chl:
            i += 2
        det_bottom = x[..., 9 + i, 0]
        opa_bottom = x[..., 10 + i, 0]

        sed1 = concentration["benthic"][..., 0]
        sed2 = concentration["benthic"][..., 1]
        sed3 = concentration["benthic"][..., 2]

        temp_bottom = dynamic_inputs["temp"][..., 0]
        taub = dynamic_inputs["taub"]

        bioom1 = self.mask_func(
            oxy_bottom,
            0.1 / 86400 * torch.exp(temp_bottom * 0.11) * oxy_bottom / (0.1 * self.redf[6] + oxy_bottom),
            0.0,
        )
        bioom5 = self.mask_func(no3_bottom, self.mask_func(oxy_bottom, 0.0, 5.0), 0.0)
        bioom6 = self.mask_func(oxy_bottom, 1.0, 0.0)
        bioom7 = self.mask_func(no3_bottom, 0.0, self.mask_func(oxy_bottom, 0.0, 1.0))

        # ----citical bottom shear stress
        Rsd = self.mask_func(
            self.parameters["crBotStr"] - taub,
            0.0,
            torch.clamp_max(
                self.parameters["resuspRt"] * ((taub**3) / (0.1 + taub**3)), self.parameters["resuspRt"]
            ),
        )
        Rds = self.mask_func(self.parameters["crBotStr"] - taub, self.parameters["sedimRt"], 0.0)

        # ---------------------------------------------------------------
        # ----denitrification parameter in dependence of available oxygen
        Rsa = self.mask_func(
            oxy_bottom, self.parameters["reminSED"] * torch.exp(self.parameters["TctrlDenit"] * temp_bottom), 0.0
        )
        Rsdenit = self.mask_func(
            oxy_bottom, 0.0, 2.0 * self.parameters["reminSED"] * torch.exp(self.parameters["TctrlDenit"] * temp_bottom)
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

        # if self.couple_co2:
        #     self.redf[15] * (Rsdenit + 2 * Rsa) * sed1
        #     self.redf[15] * self.redf[10] * ((Rsdenit + Rsa + bioom5 * Rsdenit) * sed1) - 0.5 * flux * (1.0 - bioom6)
        # --try out for phosphate ute 2.6.2010
        Rsa_p = 2.0 * self.parameters["reminSED"] * torch.exp(self.parameters["TctrlDenit"] * temp_bottom)

        yt2 = oxy_bottom / 375.0  # normalize the value as in Neumann et al 2002
        yt1 = yt2**2.0 / (self.parameters["RelSEDp2"] ** 2 + yt2**2)

        fl_pho = self.mask_func(oxy_bottom, Rsa_p * (1.0 - self.parameters["RelSEDp1"] * yt1) * sed3, Rsa_p * sed3)

        # --sed 3 phosphate pool sediment+remineralization-P release
        d_sed3 = self.mask_func(
            oxy_bottom,
            2.0 * Rsa * sed1 - Rsa_p * (1.0 - self.parameters["RelSEDp1"] * yt1) * sed3,
            Rsdenit * sed1 - Rsa_p * sed3,
        )

        # sediment opal(Si)
        d_sed2 = (
            Rds * opa_bottom
            - Rsd * sed2
            - self.parameters["reminSEDsi"] * sed2
            # - (self.parameters["burialRt"]*1000.0*(sed2**3/(sed2**3 + 1e+12)))*sed2
            - self.parameters["burialRt"] * sed2
        )
        fl_opa = Rsd * sed2 - Rds * opa_bottom

        fl_sil = self.parameters["reminSEDsi"] * sed2

        fl_zero = torch.full_like(fl_no3, 0.0)

        fl_x_list = [fl_no3, fl_nh4, fl_pho, fl_sil, fl_oxy, fl_zero, fl_zero]
        if self.use_cyanos:
            fl_x_list.append(fl_zero)
            if self.use_chl:
                fl_x_list.append(fl_zero)
        if self.use_chl:
            fl_x_list.extend([fl_zero, fl_zero])
        fl_x_list.extend([fl_zero, fl_zero, fl_det, fl_opa, fl_zero])
        fl_x = torch.stack(fl_x_list, axis=-1).unsqueeze(-1)

        d_benthic = torch.stack([d_sed1, d_sed2, d_sed3], axis=-1)

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

        oxy_surface = x[..., 4, -1]

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
        fl_sil = torch.full_like(fl_oxy, self.parameters["surface_deposition_sil"])
        fl_zero = torch.full_like(fl_oxy, 0.0)

        if self.use_cyanos:
            if self.use_chl:
                num_var = 16
            else:
                num_var = 13
        else:
            if self.use_chl:
                num_var = 14
            else:
                num_var = 12

        fl_x_list = [fl_no3, fl_nh4, fl_pho, fl_sil, fl_oxy] + [fl_zero] * (num_var - 5)

        fl_x = torch.stack(fl_x_list, axis=-1).unsqueeze(-1)

        if isinstance(concentration, TensorDict):
            return TensorDict({"pelagic": fl_x}, batch_size=concentration.batch_size)
        else:
            return fl_x
