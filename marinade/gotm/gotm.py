from typing import Optional, Union

import netCDF4 as nc
import numpy as np
import torch
import yaml
from marinade.fabm.base_model import BaseModel
from marinade.fabm.passive_model import PassiveModel
from marinade.gotm.advection import AdvectionOperator
from marinade.gotm.diffusion import DiffusionOperator
from tensordict import TensorDict


def initialize_gotm(
    gotm_yaml_path: str = "gotm.yaml",
    fabm_yaml_path: str = "fabm.yaml",
    gotm_nc_path: str = "output.nc",
    model_classes=[],
    model_arguments: list = None,
    fields=[],
    device: str = "cpu",
    dtype: str = "float64",
) -> dict:
    """Setup GOTM-FABM tracer transport simulation from configuration and output files. Currently assumes that a
    PassiveModel with a constant sinking velocity is used.

    Args:
        gotm_yaml_path (str, optional): Path to the GOTM configuration file. Defaults to 'gotm.yaml'.
        fabm_yaml_path (str, optional): Path to the FABM configuration file. Defaults to 'fabm.yaml'.
        gotm_nc_path (str, optional): Path to the GOTM/FABM NetCDF output. Defaults to 'output.nc'.
        model_classes (list, optional): List of classes that should be initialized from the .yaml files.
            Defaults to [].
        model_arguments (list of dicts, optional): Arguments for the models. If provided, has to contain as many
            dicts as model classes. Defaults to None.
        fields (list, optional): List of fields to be extracted from the dataset. Defaults to [].
        device (str, optional): The device where the tensors should be created. Defaults to 'cpu'.
        dtype (str, optional): The data type of the tensors to be created. Defaults to 'float64'.

    Returns:
        dict: Dict with the NPZD model, lists and tensors required for integrate_gotm().
    """

    dtype_class = getattr(torch, dtype)  # should give e.g. torch.float64

    if model_arguments is None:
        model_arguments = [dict() for m in model_classes]
    else:
        assert len(model_classes) == len(model_arguments), "Provide either no arguments or as many as model classes"
        for arg_dict in model_arguments:
            assert isinstance(arg_dict, dict), "Provide either no arguments or one argument dict per model class"

    # Load the config files and save number of levels and advection scheme:
    with open(gotm_yaml_path, "r") as file_gotm:
        config_gotm = yaml.safe_load(file_gotm)
        try:
            dt = config_gotm["time"]["dt"]
            if dt is None or not isinstance(dt, (int, float)):  # sanity check
                raise ValueError("Invalid dt")
            dt = torch.tensor(dt, dtype=dtype_class, device=device)
        except (KeyError, ValueError, TypeError):
            raise ValueError(
                f"time/dt key does not exist or is invalid in {gotm_yaml_path}! "
                + "Please specify a time step of type float or int."
            )
        try:
            nlev = config_gotm["grid"]["nlev"]
            if nlev is None or not isinstance(nlev, int):  # sanity check
                raise ValueError("Invalid nlev")
        except (KeyError, ValueError, TypeError):
            raise ValueError(
                f"grid/nlev key does not exist or is invalid in {gotm_yaml_path}! "
                + "Please specify a number of vertical layers as positive int."
            )
        try:
            scheme = config_gotm["fabm"]["numerics"]["w_adv_discr"]
            if isinstance(scheme, int):  # provide compatibility with Deborah's version of GOTM
                _dict = {1: "upstream", 3: "p2", 4: "superbee", 5: "muscl", 6: "p2_pdm"}
                scheme = _dict[scheme]
            if scheme not in ("upstream", "p2", "superbee", "muscl", "p2_pdm"):
                raise ValueError(
                    f"Invalid advection limiter scheme in {gotm_yaml_path}! "
                    + "Please specify fabm/numerics/w_adv_discr as one of the following: "
                    + '("upstream", "p2", "superbee", "muscl", "p2_pdm")'
                )
        except (KeyError, ValueError, TypeError):
            raise ValueError(
                f"Invalid advection limiter scheme in {gotm_yaml_path}! "
                + "Please specify fabm/numerics/w_adv_discr as one of the following: "
                + '("upstream", "p2", "superbee", "muscl", "p2_pdm")'
            )
        dilution = None
        freshwater_impact = False
        try:
            freshwater_impact = config_gotm["fabm"]["freshwater_impact"]
        except (KeyError, ValueError, TypeError):
            raise ValueError(
                f"Invalid freshwater impact specification in {gotm_yaml_path}! "
                + "Please specify fabm/freshwater_impact as true or false"
            )

    # Create the NPZD model from the files:
    models = [
        model_class.create_model_from_yaml_file(
            path_fabm=fabm_yaml_path, path_gotm=gotm_yaml_path, device=device, dtype=dtype_class, **args
        )
        for model_class, args in zip(model_classes, model_arguments)
    ]

    try:
        model = PassiveModel.create_model_from_yaml_file(
            path_fabm=fabm_yaml_path, path_gotm=gotm_yaml_path, device=device, dtype=dtype_class
        )
    except (ValueError, AttributeError, IndexError):
        model = None

    ds = nc.Dataset(gotm_nc_path, dtype=dtype)  # load the GOTM output dataset

    # The dimensions. Important for plotting the results.
    t = torch.from_numpy(ds["time"][:]).to(dtype=dtype_class, device=device)
    z = torch.from_numpy(ds["z"][:].data.squeeze()[:]).to(dtype=dtype_class, device=device)
    zi = torch.from_numpy(ds["zi"][:].data.squeeze()[:]).to(dtype=dtype_class, device=device)

    h = (
        torch.from_numpy(ds["h"][:].data.squeeze()).to(dtype=dtype_class, device=device).unsqueeze(0).unsqueeze_(-2)
    )  # shape (1 (batch), t, 1 (tracer), z)
    nuh = (
        torch.from_numpy(ds["nuh"][:].data.squeeze()).to(dtype=dtype_class, device=device).unsqueeze(0).unsqueeze_(-2)
    )  # shape (1 (batch), t, 1 (tracer), z)

    # read precipitation and evaporation if enabled
    if freshwater_impact:
        try:
            precip = torch.from_numpy(ds["precip"][:].data.squeeze()[:]).to(dtype=dtype_class, device=device)
            evap = torch.from_numpy(ds["evap"][:].data.squeeze()[:]).to(dtype=dtype_class, device=device)
            dilution = (precip + evap).unsqueeze_(0).unsqueeze_(-1)
        except (KeyError, ValueError, TypeError):
            raise ValueError("Please make sure 'precip' and 'evap' are provided in the dataset")

    # Read the initial states from the NetCDF GOTM output file.
    # Might be replaced by initialization from .yaml files.
    init_state = []
    # Important: Only pelagic variables (with a vertical size = nlev) should be included here.
    # Benthic variables can't be included here because they have a different shape
    # in the vertical dimension.
    if model is not None:
        for _variable in model.tracer_names:
            init_state.append(
                torch.from_numpy(np.array(ds[_variable.replace("/", "_")][0]))
                .squeeze()
                .to(device=device, dtype=dtype_class)
            )
        init_state = torch.stack(init_state, dim=0).unsqueeze(0)  # shape [batch, tracer, z]

    # Retrieve initial states and full time series for model tracers
    init_state_tensors = []
    init_state_benthic_tensors = []
    full_ts_tensors = []
    full_ts_benthic_tensors = []
    for _m in models:
        _i = []
        _i_b = []
        _ts = []
        _ts_b = []
        for _variable in _m.tracer_names:
            _i.append(
                torch.from_numpy(np.array(ds[_variable.replace("/", "_")][0]))
                .squeeze()
                .to(device=device, dtype=dtype_class)
            )
            _ts.append(
                torch.from_numpy(np.array(ds[_variable.replace("/", "_")][:]))
                .squeeze()
                .to(device=device, dtype=dtype_class)
            )
        for _variable in _m.benthic_tracer_names:
            _i_b.append(
                torch.from_numpy(np.array(ds[_variable.replace("/", "_")][0]))
                .squeeze()
                .to(device=device, dtype=dtype_class)
            )
            _ts_b.append(
                torch.from_numpy(np.array(ds[_variable.replace("/", "_")][:]))
                .squeeze()
                .to(device=device, dtype=dtype_class)
            )
        init_state_tensors.append(torch.stack(_i, dim=0).unsqueeze_(0) if _i else None)
        full_ts_tensors.append(torch.stack(_ts, dim=1).unsqueeze_(0) if _ts else None)
        init_state_benthic_tensors.append(torch.stack(_i_b, dim=0).unsqueeze_(0) if _i_b else None)
        full_ts_benthic_tensors.append(torch.stack(_ts_b, dim=1).unsqueeze_(0) if _ts_b else None)

    # Retrieve additional output time series for model tracers
    output_fields = []
    for _m in models:
        _o = TensorDict()
        for _variable in _m.output_fields:
            try:
                _o[_variable] = (
                    torch.from_numpy(np.array(ds[_variable.replace("/", "_")][:]))
                    .squeeze()
                    .to(device=device, dtype=dtype_class)
                )
            except (KeyError, ValueError, TypeError, IndexError):
                pass
        output_fields.append(_o if len(_o.keys()) > 0 else None)

    fields_dict = {}
    for _variable in fields:
        try:
            fields_dict[_variable] = (
                torch.from_numpy(np.array(ds[_variable][:])).squeeze().to(device=device, dtype=dtype_class)
            )
        except (KeyError, ValueError, TypeError):
            pass

    # Get dynamic_inputs for models
    dynamic_input_names = set(["h", "nu"])
    for _m in models:
        dynamic_input_names.update(_m.dependencies)
    dynamic_inputs = TensorDict(batch_size=h.shape[:2], device=h.device)
    for dependency in dynamic_input_names:
        if dependency == "wind":
            # wind is typically not provided, but needs to be computed from u10 and v10
            u10 = torch.tensor(np.array(ds["u10"]), device=h.device, dtype=h.dtype).squeeze().unsqueeze(0)
            v10 = torch.tensor(np.array(ds["v10"]), device=h.device, dtype=h.dtype).squeeze().unsqueeze(0)
            wind = torch.sqrt(u10**2 + v10**2)
            dynamic_inputs["wind"] = wind
        elif dependency == "h":
            dynamic_inputs["h"] = h
        elif dependency == "nu":
            dynamic_inputs["nu"] = nuh
        else:
            _i = torch.tensor(np.array(ds[dependency]), device=h.device, dtype=h.dtype).squeeze().unsqueeze(0)
            dynamic_inputs[dependency] = _i

    # Return the result as a dictionary so that additional results can be included in the future
    result = {
        "nlev": nlev,
        "dt": dt,
        "model": model,
        "models": models,
        "h_batches_tracers": h,
        "h": h,
        "nuh_batches_tracers": nuh,
        "nu": nuh,
        "dilution": dilution,
        "dynamic_inputs": dynamic_inputs,
        "init_state": init_state,
        "init_state_tensors": init_state_tensors,
        "full_ts_tensors": full_ts_tensors,
        "init_state_benthic_tensors": init_state_benthic_tensors,
        "full_ts_benthic_tensors": full_ts_benthic_tensors,
        "output_fields": output_fields,
        "fields": fields_dict,
        "t": t,  # for plotting purposes
        "z": z,
        "zi": zi,
        "scheme": scheme,
    }

    del ds
    return result


def run_gotm(
    gotm_yaml_path: str,
    fabm_yaml_path: str,
    gotm_nc_path: str,
    device: str = "cpu",
    dtype: str = "float64",
) -> torch.Tensor:
    """Set up and run a GOTM-FABM simulation from GOTM/FABM configuration (.yaml) and GOTM-FABM output NetCDF files.

    Computes tracer advection and diffusion given the flow fields from the GOTM simulation.

    May be used for checking consistency with the GOTM-FABM simulation in Fortran.

    Args:
        gotm_yaml_path (str): Path of the gotm.yaml configuration file.
        fabm_yaml_path (str): Path of the fabm.yaml configuration file.
        gotm_nc_path (str): Path of the NetCDF GOTM-FABM output file.
        device (str, optional): Device for torch computations. Defaults to 'cpu'.
        dtype (str, optional): Data type for torch computations. Defaults to 'float64'.

    Returns:
        torch.Tensor: Simulated tracer fields. [batch, time, tracer, nlev]
    """

    initialization_dict = initialize_gotm(gotm_yaml_path, fabm_yaml_path, gotm_nc_path, device=device, dtype=dtype)
    model: BaseModel = initialization_dict["model"]
    h_batches_tracers = initialization_dict["h_batches_tracers"]
    nuh_batches_tracers = initialization_dict["nuh_batches_tracers"]
    init_state = initialization_dict["init_state"]
    dt = initialization_dict["dt"]
    scheme = initialization_dict["scheme"]

    transport_operator = TracerSimulationOperator(
        bc_up_type_dif="neumann",
        bc_down_type_dif="neumann",
        bc_up_value_dif=0.0,
        bc_down_value_dif=0.0,
        bc_up_type_adv="flux",
        bc_down_type_adv="flux",
        bc_up_value_adv=0.0,
        bc_down_value_adv=0.0,
        implicitness=1.0,
        allow_negative=False,
        scheme_adv=scheme,
        conserve=False,
        model=model,
    )

    simulation = transport_operator.integrate(init_state, dt, h_batches_tracers, nuh_batches_tracers)

    # simulation = integrate_gotm(model, init_state, h_batches_tracers, nuh_batches_tracers, scheme=scheme)

    return simulation


class TracerSimulationOperator:
    """Simulation operator class for combined tracer advection, diffusion and BGC reactions.

    Should be initialized with the important numerical parameters and schemes for the simulation. Has a *forward* method
    for computing a single time step. Calling the operator object itself also calls its *forward* method.

    Also has an *integrate* method for a complete rollout.

    Args:
        bc_up_type_dif (str): Type of upper boundary condition for diffusion. "neumann" or "dirichlet".
        bc_down_type_dif (str): Type of lower boundary condition for diffusion. "neumann" or "dirichlet".
        bc_up_value_dif (Union[float, torch.Tensor]): Value of upper boundary condition for diffusion.
        bc_down_value_dif (Union[float, torch.Tensor]): Value of lower boundary condition for diffusion.
        bc_up_type_adv (str): Type of upper boundary condition for advection
            ("flux", "value", "oneSided", "zeroDivergence").
        bc_down_type_adv (str): Type of lower boundary condition for advection
            ("flux", "value", "oneSided", "zeroDivergence").
        bc_up_value_adv (torch.Tensor): Value for upper boundary condition for advection.
        bc_down_value_adv (torch.Tensor): Value for lower boundary condition for advection.
        scheme_adv (str): Advection limiter scheme to use ("superbee", "p2_pdm", "splmax13",
            "muscl", "central", "p2").
        conserve (bool, optional): Whether to use the conservative form of the advection equation for the
            advection due to fluid motion (sinking is always conservative). Defaults to False.
        implicitness (float, optional): Implicitness parameter. Defaults to 1.0.
        allow_negative (bool, optional): Allow negative tracer concentration values. Defaults to False.
        freshwater_impact (bool, optional): Whether to consider the impact of freshwater dilution. Defaults to False.
        method_tridiag (str, optional): Method for tridiagonal solver. One of ("thomas", "paracr", "linalg").
            Defaults to "thomas".
        implicit_backward_tridiag (bool, optional): If True, uses custom backward method based on implicit
            differentiation. Defaults to True.
        model (NPZDModel, optional): Nutrient-phytoplankton-zooplankton-detritus model for biogeochemical transport.
            If None, no biogeochemical processes (including sinking) are evaluated. Defaults to None.
        device (str, optional): Device for torch computations. Defaults to 'cpu'.
        dtype (str, optional): Data type for torch computations. Defaults to 'float64'.
    """

    def __init__(
        self,
        bc_up_type_dif: str,
        bc_down_type_dif: str,
        bc_up_value_dif: Union[float, torch.Tensor],
        bc_down_value_dif: Union[float, torch.Tensor],
        bc_up_type_adv: str,
        bc_down_type_adv: str,
        bc_up_value_adv: torch.Tensor,
        bc_down_value_adv: torch.Tensor,
        scheme_adv: str,
        conserve: bool = False,
        implicitness: float = 1.0,
        allow_negative: bool = False,
        freshwater_impact: bool = False,
        method_tridiag: str = "thomas",
        implicit_backward_tridiag: bool = True,
        model: Optional[BaseModel] = None,
        device: str = "cpu",
        dtype="float64",
        *args,
        **kwargs,
    ):
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.model = model
        self.freshwater_impact = freshwater_impact
        self.dif_op = DiffusionOperator(
            bc_up_type_dif,
            bc_down_type_dif,
            bc_up_value_dif
            if isinstance(bc_up_value_dif, torch.Tensor)
            else torch.tensor(bc_up_value_dif, device=device, dtype=dtype),
            bc_down_value_dif
            if isinstance(bc_down_value_dif, torch.Tensor)
            else torch.tensor(bc_down_value_dif, device=device, dtype=dtype),
            implicitness,
            allow_negative,
            method_tridiag,
            implicit_backward_tridiag,
        )
        if model is not None:
            self.adv_op_sinking = AdvectionOperator(
                bc_up_type_adv, bc_down_type_adv, bc_up_value_adv, bc_down_value_adv, scheme_adv, True
            )
        self.adv_op_fluid = AdvectionOperator(
            bc_up_type_adv, bc_down_type_adv, bc_up_value_adv, bc_down_value_adv, scheme_adv, conserve
        )

    def forward(
        self,
        concentration: Union[torch.Tensor, TensorDict],
        dt: Union[torch.Tensor, float],
        h: torch.Tensor,
        nu: torch.Tensor,
        bc_up_value: Optional[torch.Tensor] = None,
        bc_down_value: Optional[torch.Tensor] = None,
        lsource: Optional[torch.Tensor] = None,
        qsource: Optional[torch.Tensor] = None,
        w: Optional[torch.Tensor] = None,
        dilution: Optional[torch.Tensor] = None,
        dynamic_inputs: Optional[TensorDict] = None,
        *args,
        **kwargs,
    ):
        """Forward function for time stepping operator.

        Args:
            concentration (Union[torch.Tensor, TensorDict]): The initial tracer concentration field.
                Can be a TensorDict. Then this operator acts on *concentration["pelagic"]*. This
                functionality exists because benthic (sediment) variables might be part of the tracer
                state, but they are only on the ground and do not advect or diffuse. [batch, tracer, nlev]
            dt (Union[float, torch.Tensor]): The time step for the simulation
            h (torch.Tensor): Layer heights. [batch, 1, nlev]
            nu (torch.Tensor): Diffusivity field. [batch, (tracer or 1), nlev+1]
            bc_up_value (torch.Tensor, optional): Value of upper boundary condition. If None,
                defaults to the value set at initialization. Defaults to None.
            bc_down_value (torch.Tensor, optional): Value of lower boundary condition. If None,
                defaults to the value set at initialization. Defaults to None.
            lsource (torch.Tensor, optional): Linear source term. Defaults to None. [batch, (tracer or 1), nlev]
            qsource (torch.Tensor, optional): Constant source term. Defaults to None. [batch, (tracer or 1), nlev]
            w (torch.Tensor, optional): Vertical fluid velocity. Defaults to None. [batch, (tracer or 1), nlev+1]
            dilution (torch.Tensor, optional): Freshwater influx due to precipitation (positive) or evaporation
                (negative). Has to be provided if *freshwater_impact* was set to true at initialization of this
                operator. Defaults to None. [batch, 1]
            dynamic_inputs (TensorDict, optional): Additional inputs for the biogeochemical time step, if defined.
                Should have the shared batch size [batch]. Defaults to None.

        Returns:
            Union[torch.Tensor, TensorDict]: The final tracer concentration field. [batch, tracer, nlev]
        """
        if self.freshwater_impact:
            assert isinstance(dilution, torch.Tensor), "the input 'dilution' has to be provided as a tensor"
            sfl = concentration["pelagic"][..., -1] if isinstance(concentration, TensorDict) else concentration[..., -1]
            sfl = -sfl * dilution
            # sfl = -dilution
            bc_up_value = sfl if bc_up_value is None else sfl + bc_up_value
        # advection due to sinking
        if self.model is not None:
            ws = self.model.get_vertical_movement(concentration, dynamic_inputs)
            concentration = self.adv_op_sinking.forward(concentration, dt, h, ws)
        # advection due to fluid motion
        if w is not None:
            concentration = self.adv_op_fluid.forward(concentration, dt, h, w)
        concentration = self.dif_op.forward(concentration, dt, h, nu, lsource, qsource, bc_up_value, bc_down_value)
        if self.model is not None:
            # correct state (only changes state if model.repair_state == True)
            concentration = self.model.do_repair_state(concentration)
            concentration = self.model.forward(
                concentration,
                dt,
                dynamic_inputs=dynamic_inputs,
                h=h,
                *args,
                **kwargs,
            )
            concentration = self.model.do_repair_state(concentration)
        return concentration

    __call__ = forward

    def integrate(
        self,
        init_state: Union[torch.Tensor, TensorDict],
        dt: float,
        h: torch.Tensor,
        nu: torch.Tensor,
        bc_up_value: Optional[torch.Tensor] = None,
        bc_down_value: Optional[torch.Tensor] = None,
        lsource: Optional[torch.Tensor] = None,
        qsource: Optional[torch.Tensor] = None,
        w: Optional[torch.Tensor] = None,
        dilution: Optional[torch.Tensor] = None,
        dynamic_inputs: Optional[TensorDict] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Compute a rollout of the transport operator from initial state and forcing conditions. The shape of the
        forcing fields in the time axis determines the rollout length.

        Note that :math:`h^{n+1}`, :math:`\\nu^n`, and :math:`w^n` are used
        for the computation of the concentrations at time step :math:`n+1`.

        Args:
            init_state (Union[torch.Tensor, TensorDict]): The initial state. Can be a TensorDict.
                Then this operator acts on *init_state["pelagic"]*. This functionality exists because
                benthic (sediment) variables might be part of the tracer state, but they are only on the ground
                and do not advect or diffuse. [batch, tracer, nlev]
            dt (float): The time step for the simulation
            h (torch.Tensor): The layer heights. [batch, time, 1, nlev]
            nu (torch.Tensor): The diffusivity field. [batch, time, 1, nlev+1]
            bc_up_value (torch.Tensor, optional): Value of upper boundary condition. If None,
                defaults to the value set at initialization. Defaults to None.
            bc_down_value (torch.Tensor, optional): Value of lower boundary condition. If None,
                defaults to the value set at initialization. Defaults to None.
            lsource (torch.Tensor, optional): Linear source term (optional).
                Defaults to None. [batch, time, (tracer or 1), nlev]
            qsource (torch.Tensor, optional): Constant source term (optional).
                Defaults to None. [batch, time, (tracer or 1), nlev]
            w (torch.Tensor, optional): Vertical fluid velocity component.
                Defaults to None. [batch, time, (tracer or 1), nlev+1]
            dilution (torch.Tensor, optional): Freshwater influx due to precipitation (positive) or evaporation
                (negative). Has to be provided if *freshwater_impact* was set to true at initialization of this
                operator. Defaults to None. [batch, time, 1]
            dynamic_inputs (TensorDict, optional): Additional inputs for the biogeochemical time step, if defined.
                Should have the shared batch size [batch, time].


        Returns:
            Union[torch.Tensor, TensorDict]: The integrated tracer concentrations. [batch, time, tracer, nlev]
        """

        simulation = [init_state]

        if h.shape[1] != nu.shape[1]:
            raise ValueError("h and nu do not have the same shape in the second dimension (time)")

        # exchange axes for simpler iteration
        h = h.swapaxes(0, 1)  # time first
        nu = nu.swapaxes(0, 1)
        if lsource is not None:
            lsource = lsource.swapaxes(0, 1)
        if qsource is not None:
            qsource = qsource.swapaxes(0, 1)
        if w is not None:
            w = w.swapaxes(0, 1)
        if self.freshwater_impact:
            assert isinstance(dilution, torch.Tensor), "the input 'dilution' has to be provided as a tensor"
            dilution = dilution.swapaxes(0, 1)
        if dynamic_inputs is not None:
            dynamic_inputs = dynamic_inputs.transpose(0, 1)

        for i, (_h, _nu) in enumerate(zip(h[1:], nu[:-1])):  # use layer height of next time step, but current nu
            state = simulation[-1]
            w_fluid = w[i] if w is not None else None  # TODO might want to use i+1
            state = self.forward(
                state,
                dt,
                _h,
                _nu,
                bc_up_value=bc_up_value[i] if bc_up_value is not None else None,  # TODO might want to use i+1
                bc_down_value=bc_down_value[i] if bc_down_value is not None else None,  # TODO might want to use i+1
                lsource=lsource[i] if lsource is not None else None,  # TODO might want to use i+1
                qsource=qsource[i] if qsource is not None else None,  # TODO might want to use i+1
                w=w_fluid,
                dilution=dilution[i + 1] if self.freshwater_impact else None,
                dynamic_inputs=dynamic_inputs[i + 1] if dynamic_inputs is not None else None,
                *args,
                **kwargs,
            )
            simulation.append(state)
        return torch.stack(simulation, dim=1)
