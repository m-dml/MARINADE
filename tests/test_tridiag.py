from functools import partial
from unittest import TestCase

import torch
from marinade.gotm.tridiagonal import random_tridiag_dd, tridiagonal_bmm, tridiagonal_solve
from torch.autograd.gradcheck import gradcheck


class TestTridiag(TestCase):
    def test_gradcheck(self):
        """Check gradients of tridiagonal solvers."""
        a, b, c, y = random_tridiag_dd(requires_grad=True, n_batch=2, n_rhs=2)
        input = (a, b, c, y)
        result_thomas = gradcheck(partial(tridiagonal_solve, method="thomas"), input, eps=1e-6, atol=1e-4)
        result_paracr = gradcheck(partial(tridiagonal_solve, method="paracr"), input, eps=1e-6, atol=1e-4)
        result_linalg = gradcheck(
            partial(tridiagonal_solve, method="linalg", implicit_backward=False), input, eps=1e-6, atol=1e-4
        )
        self.assertTrue(result_thomas and result_paracr and result_linalg)

    def test_explicit_implicit(self):
        """Check consistency of explicit and implicit backward pass for gradients."""
        a, b, c, y = random_tridiag_dd(requires_grad=False, n_batch=2, n_rhs=2)
        # make independent copies
        a2, b2, c2, y2 = (
            a.clone().requires_grad_(True),
            b.clone().requires_grad_(True),
            c.clone().requires_grad_(True),
            y.clone().requires_grad_(True),
        )
        a.requires_grad = b.requires_grad = c.requires_grad = y.requires_grad = True

        x = tridiagonal_solve(a, b, c, y, implicit_backward=False)  # (n_batch, n_rhs, n_rows)
        e = x[..., 5].sum()
        e.backward()
        g_implicit = [a.grad, b.grad, c.grad, y.grad]

        x2 = tridiagonal_solve(a2, b2, c2, y2, implicit_backward=True)  # (n_batch, n_rhs, n_rows)
        e2 = x2[..., 5].sum()
        e2.backward()
        g_explicit = [a2.grad, b2.grad, c2.grad, y2.grad]

        for g, g2 in zip(g_implicit, g_explicit):
            self.assertAlmostEqual(torch.abs(g - g2).max().item(), 0.0)

    def test_tridiag_accuracy(self):
        """Check the numerical accuracy of the solvers."""
        a, b, c, y = random_tridiag_dd(requires_grad=False, n_batch=2, n_rhs=2)
        x_thomas = tridiagonal_solve(a, b, c, y, method="thomas")
        x_paracr = tridiagonal_solve(a, b, c, y, method="paracr")
        x_linalg = tridiagonal_solve(a, b, c, y, method="linalg")
        yhat_thomas = tridiagonal_bmm(a, b, c, x_thomas)
        yhat_paracr = tridiagonal_bmm(a, b, c, x_paracr)
        yhat_linalg = tridiagonal_bmm(a, b, c, x_linalg)
        self.assertAlmostEqual(torch.abs(y - yhat_thomas).max().item(), 0.0)
        self.assertAlmostEqual(torch.abs(y - yhat_paracr).max().item(), 0.0)
        self.assertAlmostEqual(torch.abs(y - yhat_linalg).max().item(), 0.0)

    def test_nograd(self):
        """Check if solutions for systems that do not require gradients require gradients."""
        a, b, c, y = random_tridiag_dd(requires_grad=False, n_batch=2, n_rhs=2)
        x_thomas = tridiagonal_solve(a, b, c, y, method="thomas")
        x_paracr = tridiagonal_solve(a, b, c, y, method="paracr")
        x_linalg = tridiagonal_solve(a, b, c, y, method="linalg")
        self.assertFalse(x_thomas.requires_grad)
        self.assertFalse(x_paracr.requires_grad)
        self.assertFalse(x_linalg.requires_grad)

    def test_dtypes(self):
        """Check type consistency."""
        for dtype in [torch.float32, torch.float64]:
            for method in ("thomas", "paracr", "linalg"):
                a, b, c, y = random_tridiag_dd(requires_grad=True, dtype=dtype, n_batch=2, n_rhs=2)
                x = tridiagonal_solve(a, b, c, y, method=method)
                e = x[..., 0].sum()
                e.backward()
                for q in [x, a.grad, b.grad, c.grad, y.grad]:
                    self.assertTrue(q.dtype is dtype)
