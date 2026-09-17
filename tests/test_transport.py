from unittest import TestCase

import torch
from marinade.fabm.passive_model import PassiveModel
from marinade.gotm.gotm import TracerSimulationOperator
from tensordict import TensorDict
from torch.autograd.gradcheck import gradcheck


class TestTransport(TestCase):
    def test_transport_operator(self):
        """Unittest for combined transport operator (advection, diffusion, BGC)"""
        nlev = 12
        T = 10
        dt = 0.1
        n_tracers = 3
        transport_operator = TracerSimulationOperator(
            "neumann", "neumann", 0.0, 0.0, "flux", "flux", 0.0, 0.0, "p2_pdm", False, model=PassiveModel(-0.8)
        )

        # There are T+1 time steps (with the initial state), and at each time step, the total mass should be conserved.
        # Therefore, the derivative of the sum of the total mass over all times with respect to the initial
        # concentration at any point in space should be T + 1, thus the gradient minus (T + 1) should be zero.

        init_state = torch.stack(
            (torch.ones((n_tracers, nlev), dtype=torch.float64), torch.zeros((n_tracers, nlev), dtype=torch.float64))
        ).requires_grad_(True)
        init_state_dict = TensorDict({"pelagic": init_state}, batch_size=(init_state.shape[0]))
        h = torch.ones((2, T, 1, nlev), dtype=torch.float64)
        nu = torch.full((2, T, 1, nlev + 1), 0.1, dtype=torch.float64)
        lsource = torch.zeros((2, T, 1, nlev)).requires_grad_(True)
        qsource = None
        w = None

        rollout = transport_operator.integrate(
            init_state=init_state_dict, dt=dt, h=h, nu=nu, lsource=lsource, qsource=qsource, w=w
        )["pelagic"]
        self.assertEqual(tuple(rollout.shape), (2, T, n_tracers, nlev))
        torch.sum(rollout).backward()
        self.assertAlmostEqual(torch.sum(torch.abs(init_state.grad - T)).numpy(), 0)

        transport_operator = TracerSimulationOperator(
            "neumann", "neumann", 0.0, 0.0, "flux", "flux", 0.0, 0.0, "p2_pdm", False
        )

        init_state = torch.stack(
            (torch.ones((n_tracers, nlev), dtype=torch.float64), torch.zeros((n_tracers, nlev), dtype=torch.float64))
        ).requires_grad_(True)
        h = torch.ones((2, T, 1, nlev), dtype=torch.float64)
        nu = torch.full((2, T, 1, nlev + 1), 0.1, dtype=torch.float64)
        lsource = None
        qsource = None
        w = torch.zeros((2, T, 1, nlev + 1)).requires_grad_(True)

        rollout = transport_operator.integrate(init_state, dt, h, nu, lsource, qsource, w)
        self.assertEqual(tuple(rollout.shape), (2, T, n_tracers, nlev))
        torch.sum(rollout).backward()
        self.assertAlmostEqual(torch.sum(torch.abs(init_state.grad - T)).numpy(), 0)

        transport_operator = TracerSimulationOperator(
            "neumann", "neumann", 0.0, 0.0, "flux", "flux", 0.0, 0.0, "p2_pdm", False
        )

        init_state = torch.stack(
            (torch.ones((n_tracers, nlev), dtype=torch.float64), torch.zeros((n_tracers, nlev), dtype=torch.float64))
        ).requires_grad_(True)
        h = torch.ones((2, T, 1, nlev), dtype=torch.float64)
        nu = torch.full((2, T, 1, nlev + 1), 0.1, dtype=torch.float64)
        lsource = None
        qsource = torch.zeros((2, T, 1, nlev)).requires_grad_(True)
        w = torch.zeros((2, T, n_tracers, nlev + 1)).requires_grad_(True)

        rollout = transport_operator.integrate(init_state, dt, h, nu, lsource=lsource, qsource=qsource, w=w)
        self.assertEqual(tuple(rollout.shape), (2, T, n_tracers, nlev))
        torch.sum(rollout).backward()
        self.assertAlmostEqual(torch.sum(torch.abs(init_state.grad - T)).numpy(), 0)
        self.assertAlmostEqual(
            torch.sum(
                torch.abs(
                    qsource.grad - torch.linspace(T - 1, 0, T).unsqueeze(-1).unsqueeze(-1).unsqueeze(0) * n_tracers * dt
                )
            ).numpy(),
            0,
            places=5,
        )

    def test_gradients(self):
        nlev = 15
        T = 2
        dt = torch.tensor(0.001, dtype=torch.float64)
        n_tracers = 3
        transport_operator = TracerSimulationOperator(
            "neumann", "neumann", 0.0, 0.0, "flux", "flux", 0.0, 0.0, "p2_pdm", False, model=PassiveModel(-0.8)
        )
        """Test gradients of transport operator."""

        init_state = torch.stack(
            (
                torch.ones((n_tracers, nlev), dtype=torch.float64),
                0.5 + torch.zeros((n_tracers, nlev), dtype=torch.float64),
            )
        ).requires_grad_(True)
        h = torch.ones((2, T, 1, nlev), dtype=torch.float64)
        nu = torch.full((2, T, 1, nlev + 1), 0.1, dtype=torch.float64)
        lsource = torch.zeros((2, T, 1, nlev), dtype=torch.float64).requires_grad_(True)
        qsource = 1.0e-2 * torch.ones((2, T, 1, nlev), dtype=torch.float64).requires_grad_(True)

        self.assertTrue(
            gradcheck(
                transport_operator.integrate,
                (init_state, dt, h, nu, None, None, lsource, qsource),
                raise_exception=False,
                fast_mode=True,
                check_forward_ad=True,
            )
        )
