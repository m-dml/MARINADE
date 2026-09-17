from unittest import TestCase

import torch
from marinade.fabm.gotm_ergom_model import GOTMErgomModel
from marinade.fabm.gotm_light_model import GOTMLightModel
from marinade.fabm.gotm_npzd_model import GOTMNPZDModel
from marinade.gotm.gotm import TracerSimulationOperator
from tensordict import TensorDict


class TestGOTMModels(TestCase):
    def test_gotm_npzd(self):
        """Unittest for the GOTM NPZD model operator."""
        nlev = 15
        T = 24
        dt = 1800.0

        model = GOTMNPZDModel(device="cpu", dtype=torch.float64, ode_method="FE")

        n_tracers = len(model.tracer_names)

        transport_operator = TracerSimulationOperator(
            "neumann", "neumann", 0.0, 0.0, "flux", "flux", 0.0, 0.0, "p2_pdm", False, model=model
        )

        h = torch.ones((2, T, 1, nlev), dtype=torch.float64)
        z = torch.cumsum(h, dim=-1) - torch.sum(h, dim=-1, keepdim=True) - h / 2
        nu = torch.full((2, T, 1, nlev + 1), 0.1, dtype=torch.float64)

        t = torch.linspace(0, (T - 1) * dt, T, dtype=torch.float64)

        par0 = torch.stack(
            [
                torch.clamp_min(-50.0 * torch.cos(2 * torch.pi * t / (24 * 60 * 60)), 0.0),
                torch.clamp_min(50.0 * torch.sin(2 * torch.pi * t / (24 * 60 * 60)), 0.0),
            ],
            dim=0,
        )
        par = par0.unsqueeze(-1) * torch.exp(z[..., 0, :] / 24.0)

        dynamic_inputs = TensorDict({"par0": par0, "par": par}, batch_size=(2, T))

        init_state = torch.stack(
            (
                torch.full((2, nlev), 4.5, dtype=torch.float64),
                torch.full((2, nlev), 1.0e-5, dtype=torch.float64),
                torch.full((2, nlev), 1.0e-5, dtype=torch.float64),
                torch.full((2, nlev), 4.5, dtype=torch.float64),
            ),
            dim=-2,
        ).requires_grad_(True)

        self.assertEqual(
            n_tracers, init_state.shape[-2], "number of tracers in model.tracer_names not equal to 4 in gotm_npzd_model"
        )

        rollout = transport_operator.integrate(init_state, dt, h, nu, dynamic_inputs=dynamic_inputs)

        rollout[:, -1, :].sum().backward()

        self.assertFalse(rollout.isnan().any(), "NaN value detected in gotm_npzd_model output")
        self.assertFalse(
            init_state.grad.isnan().any(), "NaN value detected in gradient of gotm_npzd_model output w.r.t. init_state"
        )
        self.assertTrue(
            init_state.grad.nonzero().any(), "The gradient of gotm_npzd_model output w.r.t. init_state is zero"
        )

    def test_gotm_npzd_light_coupled(self):
        """Unittest for the GOTM NPZD model operator, coupled with a light model."""
        nlev = 15
        T = 24
        dt = 1800.0

        npzd_model = GOTMNPZDModel(device="cpu", dtype=torch.float64, ode_method="RK2")
        light_model = GOTMLightModel(0.58, 0.1, 24.0, device="cpu", dtype=torch.float64)

        h = torch.ones((2, T, 1, nlev), dtype=torch.float64)
        z = torch.cumsum(h, dim=-1) - torch.sum(h, dim=-1, keepdim=True) - h / 2
        z = z[..., 0, :]
        nu = torch.full((2, T, 1, nlev + 1), 0.1, dtype=torch.float64)

        t = torch.linspace(0, (T - 1) * dt, T, dtype=torch.float64)

        I_0 = torch.stack(
            [
                torch.clamp_min(-100.0 * torch.cos(2 * torch.pi * t / (24 * 60 * 60)), 0.0),
                torch.clamp_min(100.0 * torch.sin(2 * torch.pi * t / (24 * 60 * 60)), 0.0),
            ],
            dim=0,
        ).requires_grad_()

        dynamic_inputs = TensorDict({"I_0": I_0, "h": h, "z": z, "zeta": torch.zeros_like(I_0)}, batch_size=(2, T))

        init_state = torch.stack(
            (
                torch.full((2, nlev), 4.5, dtype=torch.float64),
                torch.full((2, nlev), 1.0e-5, dtype=torch.float64),
                torch.full((2, nlev), 1.0e-5, dtype=torch.float64),
                torch.full((2, nlev), 4.5, dtype=torch.float64),
            ),
            dim=-2,
        ).requires_grad_()
        init_state = TensorDict({"pelagic": init_state})
        init_state = light_model.forward(init_state, dt, dynamic_inputs[:, 0])

        coupled_model = light_model.couple(npzd_model)

        n_tracers = len(coupled_model.tracer_names)

        transport_operator = TracerSimulationOperator(
            "neumann", "neumann", 0.0, 0.0, "flux", "flux", 0.0, 0.0, "p2_pdm", False, model=coupled_model
        )

        self.assertEqual(
            n_tracers,
            init_state["pelagic"].shape[-2],
            "number of tracers in model.tracer_names not equal to 4 in gotm_npzd_model",
        )

        rollout = transport_operator.integrate(init_state, dt, h, nu, dynamic_inputs=dynamic_inputs)

        rollout["pelagic"][:, -1, :].sum().backward()

        self.assertFalse(rollout["pelagic"].isnan().any(), "NaN value detected in gotm_npzd_model output")
        self.assertFalse(
            init_state["pelagic"].grad.isnan().any(),
            "NaN value detected in gradient of gotm_npzd_model output w.r.t. init_state",
        )
        self.assertFalse(I_0.grad.isnan().any(), "NaN value detected in gradient of gotm_npzd_model output w.r.t. I_0")
        self.assertTrue(
            init_state["pelagic"].grad.nonzero().any(),
            "The gradient of gotm_npzd_model output w.r.t. init_state is zero",
        )

    def test_gotm_ergom(self):
        """Unittest for the GOTM ERGOM model operator (based on ERGOM implementation in FABM)"""
        n_batch = 2
        nlev = 15
        T = 24
        dt = 1800.0

        model = GOTMErgomModel(fluff=False, device="cpu", dtype=torch.float64, ode_method="RK4")

        n_tracers = len(model.tracer_names)

        transport_operator = TracerSimulationOperator(
            "neumann", "neumann", 0.0, 0.0, "flux", "flux", 0.0, 0.0, "p2_pdm", False, model=model
        )

        h = torch.ones((n_batch, T, 1, nlev), dtype=torch.float64)
        z = torch.cumsum(h, dim=-1) - torch.sum(h, dim=-1, keepdim=True) - h / 2
        nu = torch.full((n_batch, T, 1, nlev + 1), 0.1, dtype=torch.float64)
        temp = torch.full((n_batch, T, nlev), 20.0, dtype=torch.float64)
        salt = torch.full((n_batch, T, nlev), 15.0, dtype=torch.float64)
        wind = torch.full((n_batch, T), 15.0, dtype=torch.float64)

        t = torch.linspace(0, (T - 1) * dt, T, dtype=torch.float64)

        par0 = torch.stack(
            [
                torch.clamp_min(-50.0 * torch.cos(2 * torch.pi * t / (24 * 60 * 60)), 0.0),
                torch.clamp_min(50.0 * torch.sin(2 * torch.pi * t / (24 * 60 * 60)), 0.0),
            ],
            dim=0,
        )
        par = par0.unsqueeze(-1) * torch.exp(z[..., 0, :] / 24.0)

        dynamic_inputs = TensorDict(
            {"h": h, "par0": par0, "par": par, "temp": temp, "salt": salt, "wind": wind}, batch_size=(2, T)
        )

        init_state = torch.full((n_batch, 9, nlev), 4.5, dtype=torch.float64).requires_grad_(True)

        self.assertEqual(
            n_tracers,
            init_state.shape[-2],
            "number of tracers in model.tracer_names not equal to 9 in gotm_ergom_model",
        )

        rollout = transport_operator.integrate(init_state, dt, h, nu, dynamic_inputs=dynamic_inputs)

        rollout[:, -1, :].sum().backward()

        self.assertFalse(rollout.isnan().any(), "NaN value detected in gotm_ergom_model output")
        self.assertFalse(
            init_state.grad.isnan().any(), "NaN value detected in gradient of gotm_ergom_model output w.r.t. init_state"
        )
        self.assertTrue(
            init_state.grad.nonzero().any(), "The gradient of gotm_ergom_model output w.r.t. init_state is zero"
        )

    def test_gotm_ergom_light_coupled(self):
        """Unittest for the GOTM ERGOM model operator coupled with a light model.

        (ERGOM model: Based on ERGOM implementation in FABM)
        """
        n_batch = 2
        nlev = 15
        T = 24
        dt = 1800.0

        ergom_model = GOTMErgomModel(fluff=True, device="cpu", dtype=torch.float64, ode_method="FE")
        light_model = GOTMLightModel(0.58, 0.1, 24.0, device="cpu", dtype=torch.float64)

        h = torch.ones((n_batch, T, 1, nlev), dtype=torch.float64)
        z = torch.cumsum(h, dim=-1) - torch.sum(h, dim=-1, keepdim=True) - h / 2
        z = z[..., 0, :]
        nu = torch.full((n_batch, T, 1, nlev + 1), 0.1, dtype=torch.float64)
        temp = torch.full((n_batch, T, nlev), 20.0, dtype=torch.float64)
        salt = torch.full((n_batch, T, nlev), 15.0, dtype=torch.float64)
        wind = torch.full((n_batch, T), 15.0, dtype=torch.float64)

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
                "wind": wind,
                "taub": taub,
            },
            batch_size=(2, T),
        )

        init_state = torch.full((n_batch, 9, nlev), 4.5, dtype=torch.float64).requires_grad_(True)
        init_state_benthic = torch.full((n_batch, 1), 4.5, dtype=torch.float64).requires_grad_(True)
        init_state = TensorDict({"pelagic": init_state, "benthic": init_state_benthic})
        init_state = light_model.forward(init_state, dt, dynamic_inputs[:, 0])

        coupled_model = light_model.couple(ergom_model)

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
            init_state.grad.isnan().any(), "NaN value detected in gradient of gotm_ergom_model output w.r.t. init_state"
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
