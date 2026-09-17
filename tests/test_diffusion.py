from unittest import TestCase

import torch
from marinade.gotm.diffusion import diffuse_at_center, DiffusionOperator
from tensordict import TensorDict


class TestDiffusion(TestCase):
    def test_diffuse_at_center(self):
        """Unittest for *diffuse_at_center* (function implementation)"""
        # test input has 2 batch elements, 2 tracers and 10 levels
        cc_before = torch.arange(0.0, 1.0, 0.1, dtype=torch.float64)
        cc_before = torch.stack([cc_before, torch.zeros_like(cc_before), torch.zeros_like(cc_before)]).unsqueeze(-2)
        cc_before = torch.cat((cc_before, torch.ones_like(cc_before)), dim=1)
        cc_before[2, 1, :] = cc_before[0, 0, :]
        cc_before.requires_grad_()

        nu = torch.ones_like(cc_before[:, :, :])
        nu[:, 1, :] = 0.0
        bc_down_value = torch.tensor([[0, 1], [0, -1], [0, -1]], dtype=torch.float64)
        cc_after = diffuse_at_center(
            value=cc_before,
            dt=0.1,
            h=torch.ones_like(cc_before[:, :1, :]),
            nu=nu,
            bc_up_type="neumann",
            bc_down_type="neumann",
            bc_up_value=0.0,  # torch.zeros((1,1), dtype=torch.float64),
            bc_down_value=bc_down_value,
        )

        test_tensor = torch.zeros((3, 2, 10), dtype=torch.float64)
        test_tensor[0, 0, :] = torch.tensor(
            [
                0.00916079783,
                0.100768776118,
                0.2000645156,
                0.30000541099,
                0.400000416230,
                0.49999958377,
                0.5999945890148,
                0.699935484408,
                0.79923122389,
                0.890839202171,
            ],
            dtype=torch.float64,
        )
        test_tensor[:, 1, :] = 1.0
        test_tensor[2, 1, :] = torch.arange(0.0, 1.0, 0.1, dtype=torch.float64)
        test_tensor[0, 1, 0] = 1.1
        test_tensor[1, 1, 0] = 0.909090909  # not 0.9 because of Patankar trick

        torch.sum(cc_after).backward()
        self.assertAlmostEqual(torch.sum(torch.abs(test_tensor - cc_after)).item(), 0)
        self.assertFalse(cc_after.isnan().any())
        self.assertFalse(cc_before.grad.isnan().any())

    def test_diffusion_operator(self):
        """Test diffusion operator implementation."""
        nlev = 10
        implicitnesses = [1.0, 0.0, 0.5]
        allow_negatives = [False, True, False]
        bc_up_types = ["neumann", "neumann", "dirichlet"]
        bc_down_types = ["neumann", "dirichlet", "dirichlet"]

        for implicitness, allow_negative, bc_up_type, bc_down_type in zip(
            implicitnesses, allow_negatives, bc_up_types, bc_down_types
        ):
            dif_op = DiffusionOperator(bc_up_type, bc_down_type, 0.0, 0.0, implicitness, allow_negative)

            cc_before = torch.ones((2, 1, nlev)).requires_grad_(True)  # defined at cell centers
            h = torch.ones((2, 1, nlev))  # defined at cell centers
            nu = torch.ones((2, 1, nlev + 1))  # defined at cell faces
            result = dif_op.forward(cc_before, 0.01, h, nu)
            self.assertEqual(result.shape, cc_before.shape)
            result.flatten()[0].backward()
            self.assertFalse(cc_before.grad.isnan().any())

            cc_before = torch.ones((2, 3, nlev))
            h = torch.ones((2, 1, nlev))
            nu = torch.ones((2, 1, nlev + 1))
            result = dif_op.forward(cc_before, 0.01, h, nu)
            self.assertEqual(result.shape, cc_before.shape)

            cc_before = torch.ones((2, 3, nlev))
            cc_before = TensorDict({"pelagic": cc_before}, batch_size=(cc_before.shape[0]))
            h = torch.ones((2, 1, nlev))
            nu = torch.ones((2, 1, nlev + 1))
            result = dif_op.forward(cc_before, 0.01, h, nu)["pelagic"]
            self.assertEqual(result.shape, cc_before["pelagic"].shape)

            cc_before = torch.ones((2, 3, nlev)).requires_grad_(True)
            h = torch.ones((2, 1, nlev))
            nu = torch.ones((2, 1, nlev + 1))
            result = dif_op.forward(cc_before, 0.01, h, nu)
            self.assertEqual(result.shape, cc_before.shape)
            result.flatten()[0].backward()
            self.assertFalse(cc_before.grad.isnan().any())

            with self.assertRaises(RuntimeError):
                cc_before = torch.ones((2, 3, nlev))  # defined at cell centers
                h = torch.ones((2, 4, nlev))  # defined at cell centers
                nu = torch.ones((2, 1, nlev + 1))  # defined at cell faces
                result = dif_op.forward(cc_before, 0.01, h, nu)

            with self.assertRaises(RuntimeError):
                cc_before = torch.ones((2, 1, nlev))  # defined at cell centers
                h = torch.ones((2, 4, nlev))  # defined at cell centers
                nu = torch.ones((2, 1, nlev + 1))  # defined at cell faces
                result = dif_op.forward(cc_before, 0.01, h, nu)

        dif_op = DiffusionOperator("neumann", "neumann", -0.1, 0.0, 1.0, False)

        cc_before = torch.ones((2, 3, nlev))  # defined at cell centers
        h = torch.ones((2, 1, nlev))  # defined at cell centers
        nu = torch.ones((2, 1, nlev + 1))  # defined at cell faces
        result = dif_op.forward(cc_before, 0.01, h, nu)
        self.assertEqual(result.shape, cc_before.shape)

        dif_op = DiffusionOperator("neumann", "neumann", 0.0, -0.1, 1.0, False)

        cc_before = torch.ones((2, 3, nlev))  # defined at cell centers
        h = torch.ones((2, 1, nlev))  # defined at cell centers
        nu = torch.ones((2, 1, nlev + 1))  # defined at cell faces
        result = dif_op.forward(cc_before, 0.01, h, nu)
        self.assertEqual(result.shape, cc_before.shape)

    def test_function_operator_identical(self):
        """Compare function and object implementation of diffusion routine."""
        nlev = 10
        dt = 0.01
        diffop = DiffusionOperator("neumann", "neumann", 0.0, 0.0)
        h = torch.ones((2, 1, nlev))
        nuh = torch.ones((2, 1, nlev))
        cc_old = torch.ones((2, 3, nlev))
        cc_new = diffuse_at_center(
            cc_old, dt, h, nuh, "neumann", "neumann", torch.tensor(0.0).to(h), torch.tensor(0.0).to(h)
        )
        cc_new_test = diffop(cc_old, dt, h, nuh)
        self.assertAlmostEqual(torch.sum(torch.abs(cc_new - cc_new_test)), 0.0)
