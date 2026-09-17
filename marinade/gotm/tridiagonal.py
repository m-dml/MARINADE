import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.functional import pad


def tridiagonal_solve(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    y: torch.Tensor,
    method: str = "thomas",
    implicit_backward: bool = True,
) -> torch.Tensor:
    """Solves a batch of tridiagonal systems :math:`A x = y` with the chosen algorithm.

    Args:
        a (torch.Tensor): Lower diagonal of A. [batch_size, (1 or n_rhs), n_rows - 1]
        b (torch.Tensor): Main diagonal of A. [batch_size, (1 or n_rhs), n_rows]
        c (torch.Tensor): Upper diagonal of A. [batch_size, (1 or n_rhs), n_rows - 1]
        y (torch.Tensor): Right-hand side vector. [batch_size, n_rhs, n_rows]
        method (str): Algorithm used to solve the linear system. One of ("thomas", "paracr", "linalg").
            Defaults to "thomas".
        implicit_backward (bool): if True, uses custom backward method based on implicit differentiation. This allows us
          to discard intermediate calculations and backpropagate gradients from x to a, b, c and y based only on the
          inputs and outputs. To achieve this, we solve another linear system on the backward pass, whose l.h.s. is the
          tranpose of A. If False, we'll solve the system with standard torch(script) operators, and the backward pass
          will differentiate through every step of the tridiagonal solver algorithm. In this case all intermediate
          computations must be stored on the computation graph.

    A is n_rows x n_rows. n_rhs is the number of right-hand-side vectors y for each batch element. The coefficients
    a,b,c of the linear system can be common (broadcasted) across r.h.s. vectors, or not.

    Assumes n_rows > 2.

    Returns:
        Solution x. [batch_size, n_rhs, n_rows]
    """
    if implicit_backward:
        return TridiagonalSolve.apply(a, b, c, y, method)
    else:
        if method == "thomas":
            return tridiagonal_solve_thomas(a, b, c, y)
        elif method == "paracr":
            return tridiagonal_solve_paracr(a, b, c, y)
        elif method == "linalg":
            return tridiagonal_solve_linalg(a, b, c, y)
        else:
            raise ValueError(f"Unknown tridiagonal solver: {method}")


class TridiagonalSolve(Function):
    """Custom tridiagonal solver operator.

    Use it by calling TridiagonalSolve.apply(a, b, c, y, method).
    """

    generate_vmap_rule = True

    @staticmethod
    def forward(
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        y: torch.Tensor,
        method: str = "thomas",
    ) -> torch.Tensor:
        """Forward method of custom tridiag operator.

        Args:
            a (torch.Tensor): Lower diagonal of A. [batch_size, (1 or n_rhs), n_rows - 1]
            b (torch.Tensor): Main diagonal of A. [batch_size, (1 or n_rhs), n_rows]
            c (torch.Tensor): Upper diagonal of A. [batch_size, (1 or n_rhs), n_rows - 1]
            y (torch.Tensor): Input vectors. [batch_size, n_rhs, n_rows]
            method (str, optional): Method for tridiagonal solver. One of ("thomas", "paracr"). Defaults to "thomas".

        Returns:
            Solution x of shape [batch_size, n_rhs, n_rows].
        """
        if method == "thomas":
            return tridiagonal_solve_thomas(a, b, c, y)
        elif method == "paracr":
            return tridiagonal_solve_paracr(a, b, c, y)
        elif method == "linalg":
            return tridiagonal_solve_linalg(a, b, c, y)
        else:
            raise ValueError(f"Unknown tridiagonal solver: {method}")

    @staticmethod
    def setup_context(ctx: torch.autograd.function.FunctionCtx, inputs: tuple, output: tuple):
        """Store tensors from forward pass that are needed for backward pass.

        Args:
            ctx (torch.autograd.function.FunctionCtx): context info
            inputs (tuple): tuple of inputs to forward method
            output (tuple): outputs from forward method
        """
        a, b, c, y, method = inputs
        x = output
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(a, b, c, x)
        if torch.autograd.forward_ad._current_level >= 0:
            ctx.save_for_forward(a, b, c, x)
        ctx.method = method

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    @once_differentiable  # higher order derivatives not yet supported
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor):
        """Backward method for tridiagonal solve, computed using implicit differentiation.

        Args:
            ctx (torch.autograd.function.FunctionCtx): context information from forward pass
            grad_output (torch.Tensor): gradient of loss w.r.t. output

        Returns:
            Tuple(torch.tensor): gradients for all inputs
        """
        # grad_output has shape (batch, n_rhs, n_rows)
        if grad_output is None:
            return None, None, None, None, None
        a, b, c, x = ctx.saved_tensors
        grad_a = grad_b = grad_c = grad_y = grad_method = None

        # To compute grad_d, we solve a tridiagonal system with transposed l.h.s.
        # For a mathematical derivation, see the docs.
        if any(ctx.needs_input_grad):
            if ctx.method == "thomas":
                grad_y = tridiagonal_solve_thomas(c, b, a, grad_output)  # (batch, n_rhs, n_rows)
            elif ctx.method == "paracr":
                grad_y = tridiagonal_solve_paracr(c, b, a, grad_output)  # (batch, n_rhs, n_rows)
            elif ctx.method == "linalg":
                grad_y = tridiagonal_solve_linalg(c, b, a, grad_output)  # (batch, n_rhs, n_rows)
            else:
                raise ValueError(f"Unknown tridiagonal solver: {ctx.method}")

        if ctx.needs_input_grad[0]:
            grad_a = -grad_y[..., 1:] * x[..., :-1]
        if ctx.needs_input_grad[1]:
            grad_b = -grad_y * x  # (batch, n_rhs, n_rows, n_rows)
        if ctx.needs_input_grad[2]:
            grad_c = -grad_y[..., :-1] * x[..., 1:]

        return grad_a, grad_b, grad_c, grad_y, grad_method

    @staticmethod
    def jvp(
        ctx: torch.autograd.function.FunctionCtx,
        grad_a: torch.Tensor,
        grad_b: torch.Tensor,
        grad_c: torch.Tensor,
        grad_y: torch.Tensor,
        grad_method=None,
    ):
        """Jacobian-vector product (JVP). Used for forward-mode automatic differentiation. This method is a custom
        implementation for *TridiagonalSolve*. It uses implicit differentiation for the efficient computation of
        gradients.

        Args:
            ctx (torch.autograd.function.FunctionCtx): context information from forward pass
            grad_a (torch.Tensor): forward-mode gradient of a
            grad_b (torch.Tensor): forward-mode gradient of b
            grad_c (torch.Tensor): forward-mode gradient of c
            grad_x (torch.Tensor): forward-mode gradient of x

        Returns:
            torch.tensor: gradient for input x
        """
        a, b, c, x = ctx.saved_tensors
        method = ctx.method

        if grad_a is None and grad_b is None and grad_c is None and grad_y is None:
            return None

        # initialize rhs
        if grad_y is not None:
            rhs = grad_y.clone()
        else:
            rhs = torch.zeros_like(x)

        # main diagonal contribution
        if grad_b is not None:
            rhs -= x * grad_b

        # lower diagonal contribution
        if grad_a is not None:
            rhs[..., 1:] -= x[..., :-1] * grad_a

        # upper diagonal contribution
        if grad_c is not None:
            rhs[..., :-1] -= x[..., 1:] * grad_c

        # solve A dx = rhs
        if method == "thomas":
            dx = tridiagonal_solve_thomas(a, b, c, rhs)
        elif method == "paracr":
            dx = tridiagonal_solve_paracr(a, b, c, rhs)
        elif method == "linalg":
            dx = tridiagonal_solve_linalg(a, b, c, rhs)
        else:
            raise ValueError(f"Unknown tridiagonal solver: {method}")

        return dx


@torch.jit.script
def tridiagonal_bmm(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Tridiagonal batched matrix multiply. Multiply the tridiagonal matrix A by x to compute Ax.

    Args:
        a (torch.Tensor): Lower diagonal of A. [batch_size, (1 or n_rhs), n_rows - 1]
        b (torch.Tensor): Main diagonal of A. [batch_size, (1 or n_rhs), n_rows]
        c (torch.Tensor): Upper diagonal of A. [batch_size, (1 or n_rhs), n_rows - 1]
        x (torch.Tensor): Input vectors. [batch_size, n_rhs, n_rows]

    Returns:
        torch.Tensor: Matrix-vector product
    """
    dhat = pad(x[..., :-1] * a, (1, 0)) + x * b + pad(x[..., 1:] * c, (0, 1))
    return dhat


@torch.jit.script
def tridiagonal_solve_thomas(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Solves a batch of tridiagonal systems Ax == y with the Thomas algorithm.

    Assumes n_rows > 2.

    Args:
        a (torch.Tensor): Lower diagonal of A. [batch_size, (1 or n_rhs), n_rows - 1].
        b (torch.Tensor): Main diagonal of A. [batch_size, (1 or n_rhs), n_rows].
        c (torch.Tensor): Upper diagonal of A. [batch_size, (1 or n_rhs), n_rows - 1].
        y (torch.Tensor): Right-hand side vector. [batch_size, n_rhs, n_rows].

    n_rows is the dimensionality of the system to be solved.
    n_rhs is the number of right-hand-side vectors y for each batch element.
    The coefficients a,b,c can be broadcasted across r.h.s. vectors, or not.

    Returns:
        torch.Tensor: Solution x of shape [batch_size, n_rhs, n_rows].
    """

    batch_size, n_rhs, n_rows = y.shape
    if (
        a.shape[-2] > 1 or b.shape[-2] > 1 or c.shape[-2] > 1
    ):  # if any of a,b,c have an extra dimension, make sure all do
        s = (batch_size, n_rhs, n_rows - 1)
        a, b, c = a.expand(s), b.expand(y.shape), c.expand(s)
    # We first make a backward sweep over the rows of the linear system, eliminating c and setting b to 1.
    # Note that row i consists of a[..., i - 1], b[..., i], c[..., i] and y[..., i].
    # There is no a in row 0, and there is no c in row n_rows - 1.
    # The updated values of a and y will be stored in a_swept and y_swept.
    a_swept, y_swept, x = torch.empty_like(a), torch.empty_like(y), torch.empty_like(y)

    # divide last row by main diagonal coefficient
    a_s, y_s = a[..., -1] / b[..., -1], y[..., -1] / b[..., -1]
    a_swept[..., -1], y_swept[..., -1] = a_s, y_s

    for i in range(n_rows - 2, -1, -1):  # work back-to-front across rows, and in paralell over batch and r.h.s. vectors
        # eliminate c in row i by adding a multiple of row i + 1:
        # Note: a_swept[..., i] refers to row i + 1, since a starts on the second row.
        b_new = b[..., i] - c[..., i] * a_s  # b after eliminating c, but before scaling b to 1
        y_new = y[..., i] - c[..., i] * y_s  # y after eliminating c, but before scaling b to 1

        # scale row i so b is 1:
        y_s = y_new / b_new  # y after scaling b to 1
        y_swept[..., i] = y_s
        if i > 0:  # Note: a_swept[..., i - 1] refers to row i, since a starts on the second row.
            a_s = a[..., i - 1] / b_new  # a after scaling b to 1. skipped for first row.
            a_swept[..., i - 1] = a_s

    # now we make another sweep from front to back, and eliminate a
    x_next = y_swept[..., 0]
    x[..., 0] = x_next
    for i in range(1, n_rows):  # compute solutions recursively
        x_next = y_swept[..., i] - a_swept[..., i - 1] * x_next  # note a[..., i - 1] is in row i
        x[..., i] = x_next

    return x


def tridiagonal_solve_paracr(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Solves a batch of tridiagonal systems Ax == y with the parallel cyclic reduction algorithm.

    Reference: Section 5.5.2 in Gallopoulos, Efstratios, Bernard Philippe, and Ahmed H. Sameh.
    Parallelism in matrix computations. Dordrecht: Springer, 2016.

    Assumes n_rows > 2.

    Args:
        a (torch.Tensor): Lower diagonal of A. [batch_size, (1 or n_rhs), n_rows - 1]
        b (torch.Tensor): Main diagonal of A. [batch_size, (1 or n_rhs), n_rows]
        c (torch.Tensor): Upper diagonal of A. [batch_size, (1 or n_rhs), n_rows - 1]
        y (torch.Tensor): Right-hand side vector. [batch_size, n_rhs, n_rows]

    n_rows is the dimensionality of the system to be solved.
    n_rhs is the number of right-hand-side vectors y for each batch element.
    The coefficients a,b,c can be broadcasted across r.h.s. vectors, or not.

    Returns:
        torch.Tensor: Solution x of shape [batch_size, n_rhs, n_rows].
    """
    n = (y.shape[-1] - 1).bit_length()
    return _tridiagonal_solve_paracr_inner(a, b, c, y, n)


@torch.jit.script
def _tridiagonal_solve_paracr_inner(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    y: torch.Tensor,
    n: int,
) -> torch.Tensor:
    """JITed internal function.

    Call tridiagonal_solve_paracr_inner instead.
    """
    for j in range(n):
        s = 1 << j  # 2**j
        # construct a matrix with lower diag -p, upper diag -q and 1 on main diag
        p = a / b[..., :-s]
        q = c / b[..., s:]
        # multiply system with new matrix
        y = y - torch.nn.functional.pad(p * y[..., :-s], (s, 0)) - torch.nn.functional.pad(q * y[..., s:], (0, s))
        b = b - torch.nn.functional.pad(p * c, (s, 0)) - torch.nn.functional.pad(q * a, (0, s))
        a = -p[..., s:] * a[..., :-s]
        c = -q[..., :-s] * c[..., s:]

    x = y / b
    return x


def tridiagonal_solve_linalg(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Solves a batch of tridiagonal systems Ax == y with torch.linalg.solve.

    Assumes n_rows > 2.

    Args:
        a (torch.Tensor): Lower diagonal of A, shape (batch_size, (1 or n_rhs), n_rows - 1).
        b (torch.Tensor): Main diagonal of A, shape (batch_size, (1 or n_rhs), n_rows).
        c (torch.Tensor): Upper diagonal of A, shape (batch_size, (1 or n_rhs), n_rows - 1).
        y (torch.Tensor): Right-hand side vector, shape (batch_size, n_rhs, n_rows).

    n_rows is the dimensionality of the system to be solved.
    n_rhs is the number of right-hand-side vectors y for each batch element.
    The coefficients a,b,c can be broadcasted across r.h.s. vectors, or not.

    Returns:
        torch.Tensor: Solution x of shape [batch_size, n_rhs, n_rows].
    """
    A = torch.zeros(b.shape + (b.shape[-1],), dtype=a.dtype, device=a.device)
    A.diagonal(offset=-1, dim1=-2, dim2=-1).copy_(a)
    A.diagonal(offset=0, dim1=-2, dim2=-1).copy_(b)
    A.diagonal(offset=1, dim1=-2, dim2=-1).copy_(c)
    return torch.linalg.solve(A, y.unsqueeze(-1)).squeeze(-1)


def tridiagonal_solve_scipy(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Solves a batch of tridiagonal systems Ax == y with scipy.linalg.solve_banded. Only works for CPU tensors.

    Args:
        a (torch.Tensor): Lower diagonal of A, shape (batch_size, (1 or n_rhs), n_rows - 1).
        b (torch.Tensor): Main diagonal of A, shape (batch_size, (1 or n_rhs), n_rows).
        c (torch.Tensor): Upper diagonal of A, shape (batch_size, (1 or n_rhs), n_rows - 1).
        y (torch.Tensor): Right-hand side vector, shape (batch_size, n_rhs, n_rows).

    n_rows is the dimensionality of the system to be solved.
    n_rhs is the number of right-hand-side vectors y for each batch element.
    The coefficients a,b,c can be broadcasted across r.h.s. vectors, or not.

    Returns:
        torch.Tensor: Solution x of shape (batch_size, n_rhs, n_rows).
    """
    assert all((a.is_cpu, b.is_cpu, c.is_cpu, y.is_cpu)), "scipy only supports cpu tensors!"
    import scipy as sp

    A = torch.zeros(b.shape[:-1] + (3, b.shape[-1]), dtype=a.dtype, device=a.device)
    A[..., 2, :-1].copy_(a)
    A[..., 1, :].copy_(b)
    A[..., 0, 1:].copy_(c)
    print(A.shape, y.shape)
    x = sp.linalg.solve_banded((1, 1), A.numpy(), y.numpy(), overwrite_ab=True)
    return torch.tensor(x, dtype=b.dtype, device=b.device)


def random_tridiag_dd(
    n_batch: int = 2, n_rhs: int = 1, n_rows: int = 10, requires_grad: bool = False, dtype=torch.float64, device="cpu"
):
    """Utility function that define a random diagonally dominant tridiagonal system. Used for testing and benchmarking.

    Args:
        n_batch (int, optional): batch size. Defaults to 2.
        n_rhs (int, optional): number of r.h.s. vectors per batch element. Defaults to 1.
        n_rows (int, optional): system dimensionality. Defaults to 10.
        requires_grad (bool, optional): whether the tensors defining the system should have gradient tracking.
            Defaults to False.
        dtype (torch.dtype, optional): Torch data type for the tridiagonal system. Defaults to torch.float64.
        device (str, optional): Device on which to initialize tensors. Defaults to 'cpu'.

    Returns:
        Tuple:
            - a (torch.Tensor): lower diagonal of shape [batch, 1, n_rows - 1]
            - b (torch.Tensor): main diagonal of shape [batch, (n_rhs or 1), n_rows - 1]
            - c (torch.Tensor): upper diagonal of shape [batch, 1, n_rows - 1]
            - y (torch.Tensor): right-hand side of shape [batch, n_rhs, n_rows]
    """
    # define a random diagonally dominant tridiagonal system
    a = (torch.rand(n_batch, 1, n_rows - 1, requires_grad=False, dtype=dtype, device=device) - 0.5) / 10.0
    b = torch.ones(n_batch, 1, n_rows, requires_grad=False, dtype=dtype, device=device)
    c = (torch.rand(n_batch, 1, n_rows - 1, requires_grad=False, dtype=dtype, device=device) - 0.5) / 10.0
    y = torch.randn(n_batch, n_rhs, n_rows, requires_grad=False, dtype=dtype, device=device)

    if requires_grad:  # turn on gradient here so that a,b,c,y are leaf nodes (not dependent on something else)
        a.requires_grad = b.requires_grad = c.requires_grad = y.requires_grad = True
    return a, b, c, y
