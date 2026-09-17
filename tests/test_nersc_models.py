from unittest import TestCase

import torch
from marinade.fabm.gotm_light_model import GOTMLightModel
from marinade.fabm.nersc_ecosmo_model import NerscEcosmoModel
from marinade.fabm.nersc_ecosmo_operational_model import NerscEcosmoOperationalModel
from marinade.fabm.nersc_ecosmo_simple_model import NerscEcosmoSimpleModel
from marinade.gotm.gotm import TracerSimulationOperator
from tensordict import TensorDict


class TestNerscModels(TestCase):
    def __test_nersc_ecosmo(self, use_cyanos, use_chl):
        """Unittest for ECOSMO II (Daewel and Schrum, 2013) model operator.

        Internal function that is repeatedly called by test_nersc_ecosmo
        """
        n_batch = 2
        nlev = 15
        T = 24
        dt = 1800.0

        model = NerscEcosmoModel(
            use_cyanos=use_cyanos,
            use_fish=False,
            couple_co2=False,
            use_chl=use_chl,
            community_export=False,
            device="cpu",
            dtype=torch.float64,
            ode_method="RK2",
        )

        n_tracers = len(model.tracer_names)

        transport_operator = TracerSimulationOperator(
            "neumann", "neumann", 0.0, 0.0, "flux", "flux", 0.0, 0.0, "p2_pdm", False, model=model
        )

        h = torch.ones((n_batch, T, 1, nlev), dtype=torch.float64)
        z = torch.cumsum(h, dim=-1) - torch.sum(h, dim=-1, keepdim=True) - h / 2
        nu = torch.full((n_batch, T, 1, nlev + 1), 0.1, dtype=torch.float64)
        temp = torch.full((n_batch, T, nlev), 20.0, dtype=torch.float64)
        salt = torch.full((n_batch, T, nlev), 15.0, dtype=torch.float64)

        t = torch.linspace(0, (T - 1) * dt, T, dtype=torch.float64)

        taub = (0.05 * (1 - torch.cos(2 * torch.pi * t / (24 * 60 * 60)))).expand(n_batch, -1)
        par0 = torch.stack(
            [
                torch.clamp_min(-50.0 * torch.cos(2 * torch.pi * t / (24 * 60 * 60)), 0.0),
                torch.clamp_min(50.0 * torch.sin(2 * torch.pi * t / (24 * 60 * 60)), 0.0),
            ],
            dim=0,
        )
        par = par0.unsqueeze(-1) * torch.exp(z[..., 0, :] / 24.0)
        mean_par = torch.full_like(par, 30.0)
        mean_par[:, T // 2 :] = 50.0
        mean_surface_par = torch.full_like(par0, 110.0)
        mean_surface_par[:, T // 2 :] = 130.0

        dynamic_inputs = TensorDict(
            {
                "h": h,
                "par0": par0,
                "par": par,
                "mean_par": mean_par,
                "mean_surface_par": mean_surface_par,
                "temp": temp,
                "salt": salt,
                "taub": taub,
            },
            batch_size=(2, T),
        )

        init_state = torch.full((2, n_tracers, nlev), 1.0, dtype=torch.float64).requires_grad_()
        init_state_benthic = torch.full((n_batch, 3), 0.1, dtype=torch.float64).requires_grad_()
        init_state = TensorDict({"pelagic": init_state, "benthic": init_state_benthic})

        rollout = transport_operator.integrate(init_state, dt, h, nu, dynamic_inputs=dynamic_inputs)

        rollout["pelagic"][:, -1, :].sum().backward()

        self.assertFalse(rollout["pelagic"].isnan().any(), "NaN value detected in nersc_ecosmo_model output")
        self.assertFalse(
            init_state.grad.isnan().any(),
            "NaN value detected in gradient of nersc_ecosmo_model output w.r.t. init_state",
        )
        self.assertFalse(
            init_state_benthic.grad.isnan().any(),
            "NaN value detected in gradient of nersc_ecosmo_model output w.r.t. init_state_benthic",
        )
        self.assertTrue(
            init_state["pelagic"].grad.nonzero().any(),
            "The gradient of nersc_ecosmo_model output w.r.t. init_state is zero",
        )
        self.assertTrue(
            init_state_benthic.grad.nonzero().any(),
            "The gradient of nersc_ecosmo_model output w.r.t. init_state_benthic is zero",
        )

    def test_nersc_ecosmo(self):
        """Unittest for ECOSMO II (Daewel and Schrum, 2013) model operator."""
        for use_cyanos in (True, False):
            for use_chl in (True, False):
                self.__test_nersc_ecosmo(use_cyanos, use_chl)
        None

    def __test_nersc_ecosmo_operational(self, use_cyanos, use_chl, use_chl_in_PI_curve):
        """Unittest for ECOSMO (operational implementation from FABM) model operator.

        Internal function that is repeatedly called by test_nersc_ecosmo_operational
        """
        n_batch = 2
        nlev = 15
        T = 24
        dt = 1800.0

        kwargs = {
            "use_cyanos": use_cyanos,
            "couple_co2": False,
            "use_chl": use_chl,
            "not_0d": True,
            "use_chl_in_PI_curve": use_chl_in_PI_curve,
            "device": "cpu",
            "dtype": torch.float64,
            "ode_method": "RK2",
        }
        if use_chl_in_PI_curve and not use_chl:
            self.assertRaises(AssertionError, NerscEcosmoOperationalModel, **kwargs)
        else:
            model = NerscEcosmoOperationalModel(**kwargs)

            n_tracers = len(model.tracer_names)

            transport_operator = TracerSimulationOperator(
                "neumann", "neumann", 0.0, 0.0, "flux", "flux", 0.0, 0.0, "p2_pdm", False, model=model
            )

            h = torch.ones((n_batch, T, 1, nlev), dtype=torch.float64)
            z = torch.cumsum(h, dim=-1) - torch.sum(h, dim=-1, keepdim=True) - h / 2
            nu = torch.full((n_batch, T, 1, nlev + 1), 0.1, dtype=torch.float64)
            temp = torch.full((n_batch, T, nlev), 20.0, dtype=torch.float64)
            salt = torch.full((n_batch, T, nlev), 15.0, dtype=torch.float64)

            t = torch.linspace(0, (T - 1) * dt, T, dtype=torch.float64)

            taub = (0.05 * (1 - torch.cos(2 * torch.pi * t / (24 * 60 * 60)))).expand(n_batch, -1)
            par0 = torch.stack(
                [
                    torch.clamp_min(-50.0 * torch.cos(2 * torch.pi * t / (24 * 60 * 60)), 0.0),
                    torch.clamp_min(50.0 * torch.sin(2 * torch.pi * t / (24 * 60 * 60)), 0.0),
                ],
                dim=0,
            )
            par = par0.unsqueeze(-1) * torch.exp(z[..., 0, :] / 24.0)
            mean_par = torch.full_like(par, 30.0)
            mean_par[:, T // 2 :] = 50.0
            mean_surface_par = torch.full_like(par0, 110.0)
            mean_surface_par[:, T // 2 :] = 130.0

            dynamic_inputs = TensorDict(
                {
                    "h": h,
                    "par0": par0,
                    "par": par,
                    "mean_par": mean_par,
                    "mean_surface_par": mean_surface_par,
                    "temp": temp,
                    "salt": salt,
                    "taub": taub,
                },
                batch_size=(2, T),
            )

            init_state = torch.full((2, n_tracers, nlev), 1.0, dtype=torch.float64).requires_grad_()
            init_state_benthic = torch.full((n_batch, 3), 0.1, dtype=torch.float64).requires_grad_()
            init_state = TensorDict({"pelagic": init_state, "benthic": init_state_benthic})

            rollout = transport_operator.integrate(init_state, dt, h, nu, dynamic_inputs=dynamic_inputs)

            rollout["pelagic"][:, -1, :].sum().backward()

            self.assertFalse(
                rollout["pelagic"].isnan().any(), "NaN value detected in nersc_ecosmo_operational_model output"
            )
            self.assertFalse(
                init_state.grad.isnan().any(),
                "NaN value detected in gradient of nersc_ecosmo_operational_model output w.r.t. init_state",
            )
            self.assertFalse(
                init_state_benthic.grad.isnan().any(),
                "NaN value detected in gradient of nersc_ecosmo_operational_model output w.r.t. init_state_benthic",
            )
            self.assertTrue(
                init_state["pelagic"].grad.nonzero().any(),
                "The gradient of nersc_ecosmo_operational_model output w.r.t. init_state is zero",
            )
            self.assertTrue(
                init_state_benthic.grad.nonzero().any(),
                "The gradient of nersc_ecosmo_operational_model output w.r.t. init_state_benthic is zero",
            )

    def test_nersc_ecosmo_operational(self):
        """Unittest for ECOSMO (operational implementation from FABM) model operator."""
        for use_cyanos in (True, False):
            for use_chl in (True, False):
                for use_chl_in_PI_curve in (True, False):
                    self.__test_nersc_ecosmo_operational(use_cyanos, use_chl, use_chl_in_PI_curve)
        None

    def test_nersc_ecosmo_operational_light_coupled(self):
        """Unittest for ECOSMO (operational implementation from FABM) model operator, coupled to a light model."""
        n_batch = 2
        nlev = 15
        T = 24
        dt = 1800.0

        ecosmo_model = NerscEcosmoOperationalModel(
            use_cyanos=True,
            couple_co2=False,
            use_chl=True,
            not_0d=True,
            use_chl_in_PI_curve=True,
            device="cpu",
            dtype=torch.float64,
            ode_method="RK2",
        )
        light_model = GOTMLightModel(
            0.58, 0.1, 24.0, tempmean_method="exponential_mean", device="cpu", dtype=torch.float64
        )
        coupled_model = light_model.couple(ecosmo_model)

        h = torch.ones((n_batch, T, 1, nlev), dtype=torch.float64)
        z = torch.cumsum(h, dim=-1) - torch.sum(h, dim=-1, keepdim=True) - h / 2
        z = z[..., 0, :]
        nu = torch.full((n_batch, T, 1, nlev + 1), 0.1, dtype=torch.float64)
        temp = torch.full((n_batch, T, nlev), 20.0, dtype=torch.float64)
        salt = torch.full((n_batch, T, nlev), 15.0, dtype=torch.float64)

        t = torch.linspace(0, (T - 1) * dt, T, dtype=torch.float64)

        taub = (0.05 * (1 - torch.cos(2 * torch.pi * t / (24 * 60 * 60)))).expand(n_batch, -1)
        I_0 = torch.stack(
            [
                torch.clamp_min(-100.0 * torch.cos(2 * torch.pi * t / (24 * 60 * 60)), 0.0),
                torch.clamp_min(100.0 * torch.sin(2 * torch.pi * t / (24 * 60 * 60)), 0.0),
            ],
            dim=0,
        )

        dynamic_inputs = TensorDict(
            {
                "h": h,
                "I_0": I_0,
                "h": h,
                "z": z,
                "zeta": torch.zeros_like(I_0),
                "temp": temp,
                "salt": salt,
                "taub": taub,
            },
            batch_size=(2, T),
        )

        init_state = torch.full((2, len(ecosmo_model.tracer_names), nlev), 1.0, dtype=torch.float64).requires_grad_()
        init_state_benthic = torch.full(
            (n_batch, len(ecosmo_model.benthic_tracer_names)), 0.1, dtype=torch.float64
        ).requires_grad_()
        init_state = TensorDict({"pelagic": init_state, "benthic": init_state_benthic})
        init_state = coupled_model.compute_light(init_state, dt, dynamic_inputs[:, 0])

        n_tracers = len(coupled_model.tracer_names)

        transport_operator = TracerSimulationOperator(
            "neumann", "neumann", 0.0, 0.0, "flux", "flux", 0.0, 0.0, "p2_pdm", False, model=coupled_model
        )

        self.assertEqual(
            n_tracers,
            init_state["pelagic"].shape[-2],
            "number of tracers in model.tracer_names not equal to 4 in gotm_ergom_model",
        )

        rollout = transport_operator.integrate(init_state, dt, h, nu, dynamic_inputs=dynamic_inputs)

        rollout["pelagic"][:, -1, :].sum().backward()

        self.assertFalse(rollout["pelagic"].isnan().any(), "NaN value detected in gotm_ergom_model output")
        self.assertFalse(
            init_state["pelagic"].grad.isnan().any(),
            "NaN value detected in gradient of gotm_ergom_model output w.r.t. init_state",
        )
        self.assertFalse(
            init_state_benthic.grad.isnan().any(),
            "NaN value detected in gradient of gotm_ergom_model output w.r.t. init_state_benthic",
        )
        self.assertTrue(
            init_state["pelagic"].grad.nonzero().any(),
            "The gradient of gotm_ergom_model output w.r.t. init_state is zero",
        )
        self.assertTrue(
            init_state_benthic.grad.nonzero().any(),
            "The gradient of gotm_ergom_model output w.r.t. init_state_benthic is zero",
        )

    def test_nersc_ecosmo_simplified_light_coupled(self):
        """Unittest for ECOSMO (simplified) model operator, coupled to a light model."""
        n_batch = 2
        nlev = 15
        T = 24
        dt = 1800.0

        ecosmo_model = NerscEcosmoSimpleModel(
            device="cpu",
            dtype=torch.float64,
            ode_method="RK4",
        )
        light_model = GOTMLightModel(0.58, 0.1, 24.0, device="cpu", dtype=torch.float64)
        coupled_model = light_model.couple(ecosmo_model)

        h = torch.ones((n_batch, T, 1, nlev), dtype=torch.float64)
        z = torch.cumsum(h, dim=-1) - torch.sum(h, dim=-1, keepdim=True) - h / 2
        z = z[..., 0, :]
        nu = torch.full((n_batch, T, 1, nlev + 1), 0.1, dtype=torch.float64)
        temp = torch.full((n_batch, T, nlev), 20.0, dtype=torch.float64)
        salt = torch.full((n_batch, T, nlev), 15.0, dtype=torch.float64)

        t = torch.linspace(0, (T - 1) * dt, T, dtype=torch.float64)

        taub = (0.05 * (1 - torch.cos(2 * torch.pi * t / (24 * 60 * 60)))).expand(n_batch, -1)
        I_0 = torch.stack(
            [
                torch.clamp_min(-100.0 * torch.cos(2 * torch.pi * t / (24 * 60 * 60)), 0.0),
                torch.clamp_min(100.0 * torch.sin(2 * torch.pi * t / (24 * 60 * 60)), 0.0),
            ],
            dim=0,
        )

        dynamic_inputs = TensorDict(
            {
                "h": h,
                "I_0": I_0,
                "h": h,
                "z": z,
                "zeta": torch.zeros_like(I_0),
                "temp": temp,
                "salt": salt,
                "taub": taub,
            },
            batch_size=(2, T),
        )

        init_state = torch.full((2, len(ecosmo_model.tracer_names), nlev), 1.0, dtype=torch.float64).requires_grad_()
        init_state_benthic = torch.full(
            (n_batch, len(ecosmo_model.benthic_tracer_names)), 0.1, dtype=torch.float64
        ).requires_grad_()
        init_state = TensorDict({"pelagic": init_state, "benthic": init_state_benthic})
        init_state = coupled_model.compute_light(init_state, dt, dynamic_inputs[:, 0])

        n_tracers = len(coupled_model.tracer_names)

        transport_operator = TracerSimulationOperator(
            "neumann", "neumann", 0.0, 0.0, "flux", "flux", 0.0, 0.0, "p2_pdm", False, model=coupled_model
        )

        self.assertEqual(
            n_tracers,
            init_state["pelagic"].shape[-2],
            "number of tracers in model.tracer_names not equal to 4 in gotm_ergom_model",
        )

        rollout = transport_operator.integrate(init_state, dt, h, nu, dynamic_inputs=dynamic_inputs)

        rollout["pelagic"][:, -1, :].sum().backward()

        self.assertFalse(rollout["pelagic"].isnan().any(), "NaN value detected in gotm_ergom_model output")
        self.assertFalse(
            init_state["pelagic"].grad.isnan().any(),
            "NaN value detected in gradient of gotm_ergom_model output w.r.t. init_state",
        )
        self.assertFalse(
            init_state_benthic.grad.isnan().any(),
            "NaN value detected in gradient of gotm_ergom_model output w.r.t. init_state_benthic",
        )
        self.assertTrue(
            init_state["pelagic"].grad.nonzero().any(),
            "The gradient of gotm_ergom_model output w.r.t. init_state is zero",
        )
        self.assertTrue(
            init_state_benthic.grad.nonzero().any(),
            "The gradient of gotm_ergom_model output w.r.t. init_state_benthic is zero",
        )
