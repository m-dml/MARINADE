from unittest import TestCase

import torch
from marinade.gotm.advection import advect_at_center, AdvectionOperator


class TestAdvection(TestCase):
    def test_advect_at_center_batch(self):
        """Unittest for *advect_at_center* (function implementation)"""
        cc_before = torch.arange(0.0, 1.0, 0.1, dtype=torch.float64)
        cc_before = torch.stack([cc_before, torch.zeros_like(cc_before)]).unsqueeze(-2).requires_grad_()

        cc_after = advect_at_center(
            cc_before,
            0.1,
            torch.ones_like(cc_before),
            torch.ones_like(cc_before),
            "flux",
            "flux",
            0,
            0,
            scheme="upstream",
            conserve=True,
        )

        test_tensor = torch.tensor(
            [[0, 0.09, 0.19, 0.29, 0.39, 0.49, 0.59, 0.69, 0.79, 0.98]], dtype=torch.float64
        ).unsqueeze(-2)
        test_tensor = torch.cat([test_tensor, torch.zeros_like(test_tensor)])

        torch.sum(cc_after).backward()
        self.assertAlmostEqual(torch.sum(torch.abs(test_tensor - cc_after)).item(), 0)
        self.assertFalse(cc_before.grad.isnan().any())

    def test_advection_operator(self):
        """Test whether the advection operator outputs have the right shapes for various inputs."""
        nlev = 10
        bc_types = ["flux", "value", "oneSided", "zeroDivergence", "flux", "flux"]
        limiter_schemes = ["superbee", "p2_pdm", "splmax13", "muscl", "central", "p2"]
        conserve_options = [True, False] * 3
        for bc, limiter, conserve in zip(bc_types, limiter_schemes, conserve_options):
            adv_op = AdvectionOperator(bc, bc, 0, 0.0, limiter, conserve)

            cc_before = torch.ones((2, 1, nlev)).requires_grad_(True)  # defined at cell centers
            h = torch.ones((2, 1, nlev))  # defined at cell centers
            w = torch.ones((2, 1, nlev + 1))  # defined at cell faces
            result = adv_op.forward(cc_before, 0.01, h, w)
            self.assertEqual(result.shape, cc_before.shape)
            result.flatten()[0].backward()
            self.assertFalse(cc_before.grad.isnan().any())

            cc_before = torch.ones((2, 3, nlev))
            h = torch.ones((1, 1, nlev))
            w = torch.ones((2, 1, nlev))
            w[1] = w[1] * 100
            result = adv_op.forward(cc_before, 0.01, h, w)
            self.assertEqual(result.shape, cc_before.shape)

            cc_before = torch.ones((2, 3, nlev))
            h = torch.ones((2, 1, nlev))
            w = torch.ones((2, 1, nlev + 1))
            result = adv_op.forward(cc_before, 0.01, h, w)
            self.assertEqual(result.shape, cc_before.shape)

            cc_before = torch.ones((2, 3, nlev)).requires_grad_(True)
            h = torch.ones((2, 1, nlev))
            w = torch.ones((2, 3, 1))
            result = adv_op.forward(cc_before, 0.01, h, w)
            self.assertEqual(result.shape, cc_before.shape)
            result.flatten()[0].backward()
            self.assertFalse(cc_before.grad.isnan().any())

            with self.assertRaises(RuntimeError):
                cc_before = torch.ones((2, 3, nlev))  # defined at cell centers
                h = torch.ones((2, 4, nlev))  # defined at cell centers
                w = torch.ones((2, 1, nlev + 1))  # defined at cell faces
                result = adv_op.forward(cc_before, 0.01, h, w)

            with self.assertRaises(RuntimeError):
                cc_before = torch.ones((2, 1, nlev))  # defined at cell centers
                h = torch.ones((2, 4, nlev))  # defined at cell centers
                w = torch.ones((2, 1, nlev + 1))  # defined at cell faces
                result = adv_op.forward(cc_before, 0.01, h, w)

    def test_function_operator_identical(self):
        """Compare function and object implementation of advection routine."""
        nlev = 10
        dt = 0.01
        diffop = AdvectionOperator("flux", "flux", 0, 0, "p2_pdm", True)
        h = torch.ones((2, 1, nlev))
        w = torch.ones((2, 1, nlev))
        cc_old = torch.ones((2, 3, nlev))
        cc_new = advect_at_center(cc_old, dt, h, w, "flux", "flux", 0, 0, "p2_pdm", True)
        cc_new_test = diffop(cc_old, dt, h, w)
        self.assertAlmostEqual(torch.sum(torch.abs(cc_new - cc_new_test)), 0)
