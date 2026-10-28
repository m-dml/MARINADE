#! /gpfs/work/nimtz/miniforge3/envs/tridiag/bin/python -u
# SBATCH --job-name=npzd_profiling
# SBATCH --nodes=1
# SBATCH --ntasks-per-node=48

# import stuff

import gc
import os
import pickle
from time import perf_counter
from typing import Optional, Union

import torch
from marinade.fabm.base_model import BaseModel
from marinade.fabm.gotm_light_model import GOTMLightModel
from marinade.fabm.gotm_npzd_model import GOTMNPZDModel
from marinade.fabm.nersc_ecosmo_model import NerscEcosmoModel
from marinade.gotm.advection import AdvectionOperator
from marinade.gotm.diffusion import DiffusionOperator
from marinade.gotm.gotm import initialize_gotm
from tensordict import TensorDict


class TracerSimulationProfiler:
    """Simulation operator class for combined tracer advection, diffusion and NPZD reactions.

    Should be initialized with the important numerical parameters and schemes for the simulation. Has a *forward* method
    for computing a single time step. Calling the operator object itself also calls its *forward* method.

    Also has an *integrate* method for a complete rollout.

    Args:
        bc_up_type_dif (str): Type of upper boundary condition for diffusion. "neumann" or "dirichlet".
        bc_down_type_dif (str): Type of lower boundary condition for diffusion. "neumann" or "dirichlet".
        bc_up_value_dif (float): Value of upper boundary condition for diffusion.
        bc_down_value_dif (float): Value of lower boundary condition for diffusion.
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
        method_tridiag (str, optional): Method for tridiagonal solver. One of ["thomas", "paracr"].
            Defaults to "thomas".
        implicit_backward_tridiag (bool, optional): If True, uses custom backward method based on implicit
            differentiation. Defaults to True.
        model (NPZDModel, optional): Nutrient-phytoplankton-zooplankton-detritus model for biogeochemical
            transport. If None, no biogeochemical processes (including sinking) are evaluated.
            Defaults to None.
        device (str, optional): Device for torch computations. Defaults to 'cpu'.
        dtype (str, optional): Data type for torch computations. Defaults to 'float64'.
    """

    def __init__(
        self,
        bc_up_type_dif: str,
        bc_down_type_dif: str,
        bc_up_value_dif: torch.Tensor,
        bc_down_value_dif: torch.Tensor,
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
        dynamic_inputs_old: Optional[TensorDict] = None,
        *args,
        **kwargs,
    ):
        """Forward function for time stepping operator.

        Args:
            concentration (torch.Tensor or TensorDict): The initial tracer concentration field.
                Can be a TensorDict. Then this operator acts on *concentration["pelagic"]*.
                This functionality exists because benthic (sediment) variables might be part
                of the tracer state, but they are only on the ground and do not advect or diffuse.
                [batch, tracer, nlev]
            dt (float or torch.Tensor): The time step for the simulation
            h (torch.Tensor): Layer heights. [batch, 1, nlev]
            nu (torch.Tensor): Diffusivity field. [batch, tracer or 1, nlev+1]
            bc_up_value (torch.Tensor, optional): Value of upper boundary condition. If None,
                defaults to the value set at initialization. Defaults to None.
            bc_down_value (torch.Tensor, optional): Value of lower boundary condition. If None,
                defaults to the value set at initialization. Defaults to None.
            lsource (torch.Tensor, optional): Linear source term. Defaults to None. [batch, tracer or 1, nlev]
            qsource (torch.Tensor, optional): Constant source term. Defaults to None. [batch, tracer or 1, nlev]
            w (torch.Tensor, optional): Vertical fluid velocity. Defaults to None. [batch, tracer or 1, nlev+1]
            dilution (torch.Tensor, optional): Freshwater influx due to precipitation (positive) or evaporation
                (negative). Has to be provided if *freshwater_impact* was set to true at initialization of this
                operator. Defaults to None. [batch, 1]
            dynamic_inputs (TensorDict, optional): Additional inputs for the biogeochemical time step, if defined.
                Should have the shared batch size [batch].
            dynamic_inputs_old (TensorDict, optional): Additional biogeochemical inputs from the previous time step.
                Should have the shared batch size [batch].

        Returns:
            torch.Tensor or TensorDict: The final tracer concentration field. [batch, tracer, nlev]
        """
        if self.freshwater_impact:
            assert isinstance(dilution, torch.Tensor), "the input 'dilution' has to be provided as a tensor"
            sfl = concentration["pelagic"][..., -1] if isinstance(concentration, TensorDict) else concentration[..., -1]
            sfl = -sfl * dilution
            # sfl = -dilution
            bc_up_value = sfl if bc_up_value is None else sfl + bc_up_value
        # advection due to sinking
        if h.is_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start_event.record()
        elif h.is_cpu:
            start_time = perf_counter()
        else:
            raise Exception("no valid device detected")
        if self.model is not None:
            ws = self.model.get_vertical_movement(concentration, dynamic_inputs)
            concentration = self.adv_op_sinking.forward(concentration, dt, h, ws)
        if h.is_cuda:
            end_event.record()
            torch.cuda.synchronize()
            adv_t = start_event.elapsed_time(end_event) / 1000  # ms -> s
        else:
            adv_t = perf_counter() - start_time
        # advection due to fluid motion
        if w is not None:
            raise Exception("w has to be none. fluid motion currently not supported!")
            concentration = self.adv_op_fluid.forward(concentration, dt, h, w)
        if h.is_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start_event.record()
        else:
            start_time = perf_counter()
        concentration = self.dif_op.forward(concentration, dt, h, nu, lsource, qsource, bc_up_value, bc_down_value)
        if h.is_cuda:
            end_event.record()
            torch.cuda.synchronize()
            dif_t = start_event.elapsed_time(end_event) / 1000  # ms -> s
        else:
            dif_t = perf_counter() - start_time
        if h.is_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start_event.record()
        else:
            start_time = perf_counter()
        if self.model is not None:
            # correct state (only changes state if model.repair_state == True)
            concentration = self.model.do_repair_state(concentration)
            concentration = self.model.forward(
                concentration,
                dt,
                dynamic_inputs=dynamic_inputs,
                dynamic_inputs_old=dynamic_inputs_old,
                h=h,
                *args,
                **kwargs,
            )
            concentration = self.model.do_repair_state(concentration)
        if h.is_cuda:
            end_event.record()
            torch.cuda.synchronize()
            bgc_t = start_event.elapsed_time(end_event) / 1000  # ms -> s
        else:
            bgc_t = perf_counter() - start_time
        return concentration, (adv_t, dif_t, bgc_t)

    __call__ = forward

    def forward_no_tracking(
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
        dynamic_inputs_old: Optional[TensorDict] = None,
        *args,
        **kwargs,
    ):
        """Forward function for time stepping operator.

        Args:
            concentration (torch.Tensor or TensorDict): The initial tracer concentration field.
                Can be a TensorDict. Then this operator acts on *concentration["pelagic"]*.
                This functionality exists because benthic (sediment) variables might be part
                of the tracer state, but they are only on the ground and do not advect or diffuse.
                [batch, tracer, nlev]
            dt (float or torch.Tensor): The time step for the simulation
            h (torch.Tensor): Layer heights. [batch, 1, nlev]
            nu (torch.Tensor): Diffusivity field. [batch, tracer or 1, nlev+1]
            bc_up_value (torch.Tensor, optional): Value of upper boundary condition. If None,
                defaults to the value set at initialization. Defaults to None.
            bc_down_value (torch.Tensor, optional): Value of lower boundary condition. If None,
                defaults to the value set at initialization. Defaults to None.
            lsource (torch.Tensor, optional): Linear source term. Defaults to None. [batch, tracer or 1, nlev]
            qsource (torch.Tensor, optional): Constant source term. Defaults to None. [batch, tracer or 1, nlev]
            w (torch.Tensor, optional): Vertical fluid velocity. Defaults to None. [batch, tracer or 1, nlev+1]
            dilution (torch.Tensor, optional): Freshwater influx due to precipitation (positive) or evaporation
                (negative). Has to be provided if *freshwater_impact* was set to true at initialization of this
                operator. Defaults to None. [batch, 1]
            dynamic_inputs (TensorDict, optional): Additional inputs for the biogeochemical time step, if defined.
                Should have the shared batch size [batch].
            dynamic_inputs_old (TensorDict, optional): Additional biogeochemical inputs from the previous time step.
                Should have the shared batch size [batch].

        Returns:
            torch.Tensor or TensorDict: The final tracer concentration field. [batch, tracer, nlev]
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
                dynamic_inputs_old=dynamic_inputs_old,
                h=h,
                *args,
                **kwargs,
            )
            concentration = self.model.do_repair_state(concentration)
        return concentration

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
            init_state (torch.Tensor or TensorDict): The initial state.
                Can be a TensorDict. Then this operator acts on *init_state["pelagic"]*.
                This functionality exists because benthic (sediment) variables might be part
                of the tracer state, but they are only on the ground and do not advect or diffuse.
                [batch, tracer, nlev]
            dt (float): The time step for the simulation
            h (torch.Tensor): The layer heights. [batch, time, 1, nlev]
            nu (torch.Tensor): The diffusivity field. [batch, time, 1, nlev+1]
            bc_up_value (torch.Tensor, optional): Value of upper boundary condition. If None,
                defaults to the value set at initialization. Defaults to None.
            bc_down_value (torch.Tensor, optional): Value of lower boundary condition. If None,
                defaults to the value set at initialization. Defaults to None.
            lsource (torch.Tensor, optional): Linear source term (optional). Defaults to None.
                [batch, time, tracer or 1, nlev]
            qsource (torch.Tensor, optional): Constant source term (optional). Defaults to None.
                [batch, time, tracer or 1, nlev]
            w (torch.Tensor, optional): Vertical fluid velocity component. Defaults to None.
                [batch, time, tracer or 1, nlev+1]
            dilution (torch.Tensor, optional): Freshwater influx due to precipitation (positive) or evaporation
                (negative). Has to be provided if *freshwater_impact* was set to true at initialization of this
                operator. Defaults to None. [batch, time, 1]
            dynamic_inputs (TensorDict, optional): Additional inputs for the biogeochemical time step, if defined.
                Should have the shared batch size [batch, time].


        Returns:
            torch.Tensor or TensorDict: The integrated tracer concentrations. [batch, time, tracer, nlev]
        """

        simulation = [init_state]
        times = []

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
            if h.is_cuda:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start_event.record()
            elif h.is_cpu:
                start_time = perf_counter()
            else:
                raise Exception("no valid device detected")
            _ = self.forward_no_tracking(
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
                dynamic_inputs_old=dynamic_inputs[i] if dynamic_inputs is not None else None,
                *args,
                **kwargs,
            )
            if h.is_cuda:
                end_event.record()
                torch.cuda.synchronize()
                all_t = start_event.elapsed_time(end_event) / 1000  # ms -> s
            else:
                all_t = perf_counter() - start_time
            state, _t = self.forward(
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
                dynamic_inputs_old=dynamic_inputs[i] if dynamic_inputs is not None else None,
                *args,
                **kwargs,
            )
            all_t = all_t - sum(_t)  # overhead
            simulation.append(state)
            times.append((*_t, all_t))
        return torch.stack(simulation, dim=1), torch.tensor(times, dtype=torch.float64).T


def profile_npzd(save_path=None, device="cpu", rollout=False, requires_gradients=False, dif_method=None):
    """Measure time and memory costs of advection, diffusion and NPZD operators."""

    path = "/gpfs/work/nimtz/experiments/nns_npzd/base_simulation/"
    filename = "nns_annual.nc"
    fabm_yaml_name = "fabm.yaml"
    gotm_yaml_name = "gotm.yaml"

    print("profiling for NPZD model")
    print("device:", device)
    print("rollout:", rollout)
    print("dif_method:", dif_method)

    # set up simulation, initialize

    initialization_dict = initialize_gotm(
        path + gotm_yaml_name,
        path + fabm_yaml_name,
        path + filename,
        model_classes=[GOTMLightModel, GOTMNPZDModel],
        device=device,
        dtype="float64",
    )

    h, nu, dt = [initialization_dict[key] for key in ("h", "nu", "dt")]

    initial_state_pelagic = initialization_dict["init_state_tensors"][1]
    ts_pelagic = initialization_dict["full_ts_tensors"][1]
    reference_ts_dict = TensorDict({"pelagic": ts_pelagic}, batch_size=ts_pelagic.shape[:2])
    dynamic_inputs = initialization_dict["dynamic_inputs"]

    light_model, npzd_model = initialization_dict["models"]
    light_model.repair_state = npzd_model.repair_state = True
    coupled_model = light_model.couple(npzd_model)

    forward_operator = TracerSimulationProfiler(
        bc_up_type_dif="neumann",
        bc_down_type_dif="neumann",
        bc_up_value_dif=0.0,
        bc_down_value_dif=0.0,
        bc_up_type_adv="flux",
        bc_down_type_adv="flux",
        bc_up_value_adv=0.0,
        bc_down_value_adv=0.0,
        scheme_adv="p2_pdm",
        conserve=False,
        implicitness=1.0,
        allow_negative=False,
        model=coupled_model,
    )

    if dif_method is not None:
        forward_operator.dif_op.method = dif_method

    initial_state_dict = TensorDict({"pelagic": initial_state_pelagic}, batch_size=initial_state_pelagic.shape[:1])
    initial_state_dict = light_model.compute_light(initial_state_dict, dt, dynamic_inputs[:, 0])

    # define forward function for combined model step, including time/memory tracking – done

    # define function for complete simulation – done

    # define looping function that simulates ~10 times, plus ~3 warmup runs -> time/memory

    if rollout:
        # warmup
        ts_simulation, _ = forward_operator.integrate(
            initial_state_dict, dt, h[:, :200], nu[:, :200], dynamic_inputs=dynamic_inputs[:, :200]
        )
        assert (
            ts_simulation["pelagic"] - ts_pelagic[:, :200]
        ).abs().sum() < 1e-7, "simulation output does not match fortran simulation!"

        # if gradients ahould be tracked, set corresponding flags
        if requires_gradients:
            initial_state_dict.requires_grad_()
            forward_operator.model.update_parameters(forward_operator.model.get_parameters().requires_grad_())

        _, times = forward_operator.integrate(
            initial_state_dict,
            dt,
            h,  # [:,:2000],
            nu,  # [:,:2000],
            dynamic_inputs=dynamic_inputs,  # [:,:2000]
        )

    else:
        # warmup
        print("starting warmup...")
        for i in range(10):
            print(i)
            res, _ = forward_operator.forward(
                reference_ts_dict[0, :-1],  # h[1:], nu[:-1]
                dt,
                h[0, 1:],
                nu[0, :-1],
                dynamic_inputs=dynamic_inputs[0, 1:],
            )
            assert (
                ts_pelagic[0, 1:] - res["pelagic"]
            ).abs().sum() < 1e-7, "forward step does not accurately reproduce fortran results!"

        print("starting test runs...")
        times = []
        forward_operator.model.get_parameters().clone()
        for i in range(100):
            print(i)
            if h.is_cuda:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start_event.record()
            elif h.is_cpu:
                start_time = perf_counter()
            else:
                raise Exception("no valid device detected")
            forward_operator.forward_no_tracking(
                reference_ts_dict[0, :-1],  # h[1:], nu[:-1]
                dt,
                h[0, 1:],
                nu[0, :-1],
                dynamic_inputs=dynamic_inputs[0, 1:],
            )
            if h.is_cuda:
                end_event.record()
                torch.cuda.synchronize()
                all_t = start_event.elapsed_time(end_event) / 1000  # ms -> s
            else:
                all_t = perf_counter() - start_time

            _, _t = forward_operator.forward(
                reference_ts_dict[0, :-1],  # h[1:], nu[:-1]
                dt,
                h[0, 1:],
                nu[0, :-1],
                dynamic_inputs=dynamic_inputs[0, 1:],
            )
            all_t = all_t - sum(_t)  # overhead
            times.append((*_t, all_t))
        times = torch.tensor(times, dtype=torch.float64).T

    print(times.sum(dim=1))

    if save_path is not None:
        with open(
            save_path
            + f'_{device}_{"rollout" if rollout else "parallel"}_{"grad" if requires_gradients else "nograd"}{"" if dif_method is None else "_"+dif_method}.pkl',
            "wb",
        ) as file:
            pickle.dump(times, file)

    return times


def profile_ecosmo(save_path=None, device="cpu", rollout=False, requires_gradients=False, dif_method=None):
    """Measure time and memory costs of advection, diffusion and ECOSMO operators."""

    path = "/gpfs/work/nimtz/experiments/nns_ecosmo/base_simulation/"
    filename = "nns_annual.nc"
    fabm_yaml_name = "fabm.yaml"
    gotm_yaml_name = "gotm.yaml"

    print("profiling for Ecosmo model")
    print("device:", device)
    print("rollout:", rollout)
    print("dif_method:", dif_method)

    # set up simulation, initialize

    initialization_dict = initialize_gotm(
        path + gotm_yaml_name,
        path + fabm_yaml_name,
        path + filename,
        model_classes=[GOTMLightModel, NerscEcosmoModel],
        device=device,
        dtype="float64",
    )

    h, nu, dt = [initialization_dict[key] for key in ("h", "nu", "dt")]

    initial_state_pelagic = initialization_dict["init_state_tensors"][1]
    initial_state_benthic = initialization_dict["init_state_benthic_tensors"][1]
    ts_pelagic = initialization_dict["full_ts_tensors"][1]
    ts_benthic = initialization_dict["full_ts_benthic_tensors"][1]
    reference_ts_dict = TensorDict({"pelagic": ts_pelagic, "benthic": ts_benthic}, batch_size=ts_pelagic.shape[:2])
    dynamic_inputs = initialization_dict["dynamic_inputs"]

    light_model, npzd_model = initialization_dict["models"]
    light_model.repair_state = npzd_model.repair_state = True
    coupled_model = light_model.couple(npzd_model)

    forward_operator = TracerSimulationProfiler(
        bc_up_type_dif="neumann",
        bc_down_type_dif="neumann",
        bc_up_value_dif=0.0,
        bc_down_value_dif=0.0,
        bc_up_type_adv="flux",
        bc_down_type_adv="flux",
        bc_up_value_adv=0.0,
        bc_down_value_adv=0.0,
        scheme_adv="p2_pdm",
        conserve=False,
        implicitness=1.0,
        allow_negative=False,
        model=coupled_model,
    )

    if dif_method is not None:
        forward_operator.dif_op.method = dif_method

    initial_state_dict = TensorDict(
        {"pelagic": initial_state_pelagic, "benthic": initial_state_benthic}, batch_size=initial_state_pelagic.shape[:1]
    )
    initial_state_dict = light_model.compute_light(initial_state_dict, dt, dynamic_inputs[:, 0])

    # define forward function for combined model step, including time/memory tracking – done

    # define function for complete simulation – done

    # define looping function that simulates ~10 times, plus ~3 warmup runs -> time/memory

    if rollout:
        # warmup
        ts_simulation, _ = forward_operator.integrate(
            initial_state_dict, dt, h[:, :200], nu[:, :200], dynamic_inputs=dynamic_inputs[:, :200]
        )
        assert (
            ts_simulation["pelagic"] - ts_pelagic[:, :200]
        ).abs().sum() < 1e-6, "simulation output does not match fortran simulation (pelagic)!"
        assert (
            ts_simulation["benthic"] - ts_benthic[:, :200]
        ).abs().sum() < 1e-6, "simulation output does not match fortran simulation (benthic)!"

        # if gradients ahould be tracked, set corresponding flags
        if requires_gradients:
            initial_state_dict.requires_grad_()
            forward_operator.model.update_parameters(forward_operator.model.get_parameters().requires_grad_())

        _, times = forward_operator.integrate(
            initial_state_dict,
            dt,
            h,  # [:,:2000],
            nu,  # [:,:2000],
            dynamic_inputs=dynamic_inputs,  # [:,:2000]
        )

    else:
        # warmup
        print("starting warmup...")
        for i in range(10):
            print(i)
            res, _ = forward_operator.forward(
                reference_ts_dict[0, :-1],  # h[1:], nu[:-1]
                dt,
                h[0, 1:],
                nu[0, :-1],
                dynamic_inputs=dynamic_inputs[0, 1:],
            )
            assert (
                ts_pelagic[0, 1:] - res["pelagic"]
            ).abs().sum() < 1e-5, "forward step does not accurately reproduce fortran results (pelagic)!"
            assert (
                ts_benthic[0, 1:] - res["benthic"]
            ).abs().sum() < 1e-6, "forward step does not accurately reproduce fortran results (benthic)!"

        print("starting test runs...")
        times = []
        forward_operator.model.get_parameters().clone()
        for i in range(100):
            print(i)
            if h.is_cuda:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start_event.record()
            elif h.is_cpu:
                start_time = perf_counter()
            else:
                raise Exception("no valid device detected")
            forward_operator.forward_no_tracking(
                reference_ts_dict[0, :-1],  # h[1:], nu[:-1]
                dt,
                h[0, 1:],
                nu[0, :-1],
                dynamic_inputs=dynamic_inputs[0, 1:],
            )
            if h.is_cuda:
                end_event.record()
                torch.cuda.synchronize()
                all_t = start_event.elapsed_time(end_event) / 1000  # ms -> s
            else:
                all_t = perf_counter() - start_time

            _, _t = forward_operator.forward(
                reference_ts_dict[0, :-1],  # h[1:], nu[:-1]
                dt,
                h[0, 1:],
                nu[0, :-1],
                dynamic_inputs=dynamic_inputs[0, 1:],
            )
            all_t = all_t - sum(_t)  # overhead
            times.append((*_t, all_t))
        times = torch.tensor(times, dtype=torch.float64).T

    print(times.sum(dim=1))

    if save_path is not None:
        with open(
            save_path
            + f'_{device}_{"rollout" if rollout else "parallel"}_{"grad" if requires_gradients else "nograd"}{"" if dif_method is None else "_"+dif_method}.pkl',
            "wb",
        ) as file:
            pickle.dump(times, file)

    return times


if __name__ == "__main__":
    print("running on node", os.environ.get("SLURMD_NODENAME"))

    profile_npzd(
        "/gpfs/work/nimtz/experiments/nns_npzd/base_simulation/performance_profiling/",
        device="cuda",
        rollout=False,
        dif_method="thomas",
    )
    # profile_ecosmo(
    #     '/gpfs/work/nimtz/experiments/nns_ecosmo/base_simulation/performance_profiling/',
    #     device='cuda',
    #     rollout=False,
    #     dif_method='thomas'
    # )
