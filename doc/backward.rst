.. _backward:

Tridiagonal Backward Pass
=========================
We implement a tridiagonal solve for :math:`\mathbf{A}\mathbf{x} = \mathbf{y}`, where :math:`\mathbf{A}` is a tridiagonal matrix:

.. math::
    \mathbf{A} = \begin{bmatrix}
        b_1    & c_1    & 0      & 0      & \cdots   & 0       & 0       & 0 \\
        a_1    & b_2    & c_2    & 0      & \cdots   & 0       & 0       & 0 \\
        0      & a_2    & b_3    & 0      & \cdots   & 0       & 0       & 0 \\
        \vdots & \vdots & \vdots & \vdots & \ddots   & \vdots  & \vdots  & \vdots \\
        0      & 0      & 0      & 0      & \cdots   & a_{n-2} & b_{n-1} & c_{n-1} \\
        0      & 0      & 0      & 0      & \cdots   & 0       & a_{n-1} & b_{n} \\
    \end{bmatrix}

How do we compute gradients through the solver operation :math:`\mathbf{a}, \mathbf{b}, \mathbf{c}, \mathbf{y} \rightarrow \mathbf{x}`? Concretely, suppose we
already have gradients :math:`\frac{\mathrm{d}\mathcal L}{\mathrm{d}\mathbf{x}}` of a scalar loss :math:`\mathcal L` with respect to the output,
:math:`\mathbf{x}`. How do we then calculate input gradients
:math:`\frac{\mathrm{d}\mathcal L}{\mathrm{d}\mathbf{a}}, \frac{\mathrm{d}\mathcal L}{\mathrm{d}\mathbf{b}}, \frac{\mathrm{d}\mathcal L}{\mathrm{d}\mathbf{c}}, \frac{\mathrm{d}\mathcal L}{\mathrm{d}\mathbf{y}}`?

If we solve these equations using an algorithm in torch(script) that supports autodiff, we can of course just simply
backpropagate through each step of the algorithm. This can be done by calling
:func:`~marinade.gotm.tridiagonal.tridiagonal_solve` with `implicit_backward=False`. However, this is quite memory
inefficient, since it requires us to store all intermediate computations. For some algorithms, calculating gradients
this way could also be numerically unstable, even when the forward pass is well-behaved.

We therefore implemented a more efficient approach based on implicit differentiation as a custom torch operation, which
we derive here. Beginning with :math:`\frac{\mathrm{d}\mathcal L}{\mathrm{d}\mathbf{y}}`, we have by the chain rule:

.. math::
    \frac{\mathrm{d}\mathcal L}{\mathrm{d}\mathbf{y}} = \frac{\mathrm{d}\mathcal L}{\mathrm{d} \mathbf{x}} \frac{\mathrm{d}\mathbf{x}}{\mathrm{d}\mathbf{y}}

To obtain the unknown r.h.s. factor :math:`\frac{\mathrm{d}\mathbf{x}}{\mathrm{d}\mathbf{y}}`, we differentiate both sides of the linear system with respect
to :math:`\mathbf{y}`:

.. math::
    \frac{\mathrm{d}}{\mathrm{d}\mathbf{y}}(\mathbf{A}\mathbf{x}) &= \frac{\mathrm{d}}{\mathrm{d}\mathbf{y}}(\mathbf{y}) \\
    \frac{\mathrm{d}\mathbf{x}}{\mathrm{d}\mathbf{y}} &= \mathbf{A}^{-1}

substituting this into the chain rule above, we have:

.. math::
    \frac{\mathrm{d}\mathcal L}{\mathrm{d}\mathbf{y}} \mathbf{A} &= \frac{\mathrm{d}\mathcal L}{\mathrm{d} \mathbf{x}} \\
    \mathbf{A}^T \frac{\mathrm{d}\mathcal L}{\mathrm{d}\mathbf{y}}^T &= \frac{\mathrm{d}\mathcal L}{\mathrm{d} \mathbf{x}}^T

Thus we have a new tridiagonal system with transposed matrix that we can solve for the desired gradients
:math:`\frac{\mathrm{d}\mathcal L}{\mathrm{d}\mathbf{y}}^T`. Note that these gradients are normally defined as row vector (many inputs,
one output), but here they appear in transposed form as the unknown solution of a tridiagonal system. The r.h.s. vector is
simply the incoming gradients for the backward pass. We thus solve this system inside the custom
`~marinade.gotm.tridiagonal.TridiagonalSolve.backward()` method of the
:class:`~marinade.gotm.tridiagonal.TridiagonalSolve` class to compute :math:`\frac{\mathrm{d}\mathcal L}{\mathrm{d}\mathbf{y}}`.

To compute :math:`\frac{\mathrm{d}\mathcal L}{\mathrm{d}\mathbf{b}}`, we next differentiate the linear system w.r.t. :math:`b_i`:

.. math::
    \frac{\mathrm{d}}{\mathrm{d}b_i}(\mathbf{A}\mathbf{x}) &= \frac{\mathrm{d}}{\mathrm{d}b_i}(\mathbf{y}) \\
    \frac{\mathrm{d}\mathbf{A}}{\mathrm{d}b_i}\mathbf{x} + \mathbf{A}\frac{\mathrm{d}\mathbf{x}}{\mathrm{d}b_i} &= 0

Noting that by the definition of :math:`\mathbf{A}` we have :math:`\frac{\mathrm{d}\mathbf{A}}{\mathrm{d}b_i} = \mathbf{e}_i\mathbf{e}_i^t`, where :math:`\mathbf{e}_i` is the
:math:`i`-th basis vector, we arrive at:

.. math::
    \mathbf{A}\frac{\mathrm{d}\mathbf{x}}{\mathrm{d}b_i} &= -x_i \mathbf{e}_i \\
    \frac{\mathrm{d}\mathbf{x}}{\mathrm{d}b_i} &= -x_i \mathbf{A}^{-1} \mathbf{e}_i

Substituting this result into chain rule as before, and using the fact derived above that
:math:`\frac{\mathrm{d}\mathcal L}{\mathrm{d} \mathbf{x}}=\frac{\mathrm{d}\mathcal L}{\mathrm{d}\mathbf{y}} \mathbf{A}` gives:

.. math::
    \frac{\mathrm{d}\mathcal L}{\mathrm{d}b_i} &= \frac{\mathrm{d}\mathcal L}{\mathrm{d} \mathbf{x}} \frac{\mathrm{d}\mathbf{x}}{\mathrm{d}b_i} \\
    \frac{\mathrm{d}\mathcal L}{\mathrm{d}b_i} &= \left(\frac{\mathrm{d}\mathcal L}{\mathrm{d} \mathbf{y}} \mathbf{A}\right) \left(-x_i \mathbf{A}^{-1} \mathbf{e}_i\right) \\
    \frac{\mathrm{d}\mathcal L}{\mathrm{d}b_i} &= -x_i \left(\frac{\mathrm{d}\mathcal L}{\mathrm{d} \mathbf{y}}\right)_i

Combining this result :math:`\forall i` yields:

.. math::
    \frac{\mathrm{d}\mathcal L}{\mathrm{d}\mathbf{b}} = -\mathbf{x} \otimes \frac{\mathrm{d}\mathcal L}{\mathrm{d} \mathbf{y}}

Where :math:`\frac{\mathrm{d}\mathcal L}{\mathrm{d} \mathbf{y}}` has been computed through a linear solve above.

The same logic can be easily adapted to compute gradients for :math:`\mathbf{a}` and :math:`\mathbf{c}`. The different location of these
parameters within :math:`\mathbf{A}` gives:

.. math::
    \frac{\mathrm{d}\mathbf{A}}{\mathrm{d}a_i} & = \mathbf{e}_{i+1}\mathbf{e}_i^t \\
    \frac{\mathrm{d}\mathbf{A}}{\mathrm{d}c_i} & = \mathbf{e}_{i}\mathbf{e}_{i+1}^t

Repeating the same steps as for :math:`\mathbf{b}` above, we arrive at:

.. math::
    \frac{\mathrm{d}\mathcal L}{\mathrm{d}\mathbf{a}} &= -\mathbf{x}_{1:n-1} \otimes \left(\frac{\mathrm{d}\mathcal L}{\mathrm{d} \mathbf{y}}\right)_{2:n} \\
    \frac{\mathrm{d}\mathcal L}{\mathrm{d}\mathbf{c}} &= -\mathbf{x}_{2:n}   \otimes \left(\frac{\mathrm{d}\mathcal L}{\mathrm{d} \mathbf{y}}\right)_{1:n-1}

Where :math:`\otimes` denotes elementwise multiplication. Thus, we can carry out backpropagation for :math:`\mathbf{a},\mathbf{b},\mathbf{c},\mathbf{y}`
using only their values and that of :math:`\mathbf{x}`, without storing intermediate calculations of the forward solve. In other
words, only the calculated solution :math:`\mathbf{x}` matters, not how we arrive at it. This is the algorithm called by
:func:`~marinade.gotm.tridiagonal.tridiagonal_solve` when `implicit_backward=True`, which is the default setting.

Note that in the above we have used 1-based indexing as is common in mathematical notation, whereas the actual
implementation uses 0-based indexing. Additionally, :math:`\mathbf{a},\mathbf{b},\mathbf{c},\mathbf{y}` all have a batch dimension. The solver and its
:func:`~marinade.gotm.tridiagonal.TridiagonalSolve.backward()` method support multiple r.h.s. vectors for each batch element. Variation of :math:`\mathbf{a},\mathbf{b},\mathbf{c},\mathbf{y}` across these r.h.s.
vectors is optional.


Tridiagonal Forward Pass
========================
For forward-mode gradient computation using Jacobian-vector products (JVP), we can employ a similar scheme.
Assuming knowledge of :math:`\frac{\mathrm{d}\mathbf{a}}{\mathrm{d}\boldsymbol{\lambda}}, \frac{\mathrm{d}\mathbf{b}}{\mathrm{d}\boldsymbol{\lambda}}, \frac{\mathrm{d}\mathbf{c}}{\mathrm{d}\boldsymbol{\lambda}}, \frac{\mathrm{d}\mathbf{y}}{\mathrm{d}\boldsymbol{\lambda}}`
for some parameter :math:`\boldsymbol{\lambda}`, we find :math:`\frac{\mathrm{d}\mathbf{x}}{\mathrm{d}\boldsymbol{\lambda}}` by the multivariable chain rule.
Using Einstein summation notation, we find:

.. math::
    \frac{\mathrm{d}\mathbf{x}(\mathbf{a},\mathbf{b},\mathbf{c},\mathbf{y})}{\mathrm{d}\boldsymbol{\lambda}}
    = \frac{\mathrm{d}\mathbf{x}}{\mathrm{d} a_i}\frac{\mathrm{d}a_i}{\mathrm{d}\boldsymbol{\lambda}}
    + \frac{\mathrm{d}\mathbf{x}}{\mathrm{d} b_j}\frac{\mathrm{d}b_j}{\mathrm{d}\boldsymbol{\lambda}}
    + \frac{\mathrm{d}\mathbf{x}}{\mathrm{d} c_k}\frac{\mathrm{d}c_k}{\mathrm{d}\boldsymbol{\lambda}}
    + \frac{\mathrm{d}\mathbf{x}}{\mathrm{d} y_l}\frac{\mathrm{d}y_l}{\mathrm{d}\boldsymbol{\lambda}}

With

.. math::
    \frac{\mathrm{d}\mathbf{x}}{\mathrm{d}\mathbf{y}} &= \mathbf{A}^{-1}, \\
    \frac{\mathrm{d}\mathbf{x}}{\mathrm{d}b_j} &= -x_j\mathbf{A}^{-1}\mathbf{e}_j, \\
    \frac{\mathrm{d}\mathbf{x}}{\mathrm{d}a_i} &= -\mathbf{A}^{-1}\frac{\mathrm{d}\mathbf{A}}{\mathrm{d}a_i}\mathbf{x} \\
    &= -\mathbf{A}^{-1}\mathbf{e}_{i+1}\mathbf{e}_i^t\mathbf{x} \\
    &= -\mathbf{A}^{-1}\mathbf{e}_{i+1}x_i, \\
    \frac{\mathrm{d}\mathbf{x}}{\mathrm{d}c_k} &= -\mathbf{A}^{-1}\frac{\mathrm{d}\mathbf{A}}{\mathrm{d}c_k}\mathbf{x} \\
    &= -\mathbf{A}^{-1}\mathbf{e}_k\mathbf{e}_{k+1}^t\mathbf{x} \\
    &= -\mathbf{A}^{-1}\mathbf{e}_kx_{k+1}

as before, we find

.. math::
    \frac{\mathrm{d}\mathbf{x}}{\mathrm{d}\boldsymbol{\lambda}}
    &= -\mathbf{A}^{-1}\mathbf{e}_{i+1}x_i\frac{\mathrm{d}a_i}{\mathrm{d}\boldsymbol{\lambda}}
    - x_j\mathbf{A}^{-1}\mathbf{e}_j\frac{\mathrm{d}b_j}{\mathrm{d}\boldsymbol{\lambda}}
    - \mathbf{A}^{-1}\mathbf{e}_kx_{k+1}\frac{\mathrm{d}c_k}{\mathrm{d}\boldsymbol{\lambda}}
    + \mathbf{A}^{-1}\frac{\mathrm{d}\mathbf{y}}{\mathrm{d}\boldsymbol{\lambda}} \\
    \Rightarrow
    \mathbf{\mathbf{A}}\frac{\mathrm{d}\mathbf{x}}{\mathrm{d}\boldsymbol{\lambda}}
    &= -
    \begin{pmatrix}
        0 \\
        \mathbf{x}_{[1:n-1]}\otimes\frac{\mathrm{d}\mathbf{a}}{\mathrm{d}\boldsymbol{\lambda}}
    \end{pmatrix}
    - \mathbf{x}\otimes\frac{\mathrm{d} \mathbf{b}}{\mathrm{d}\boldsymbol{\lambda}}
    - \begin{pmatrix}
        \mathbf{x}_{[2:n]}\otimes\frac{\mathrm{d}\mathbf{c}}{\mathrm{d}\boldsymbol{\lambda}} \\
        0
    \end{pmatrix}
    + \frac{\mathrm{d}\mathbf{y}}{\mathrm{d}\boldsymbol{\lambda}}.

Again, this system can be solved for :math:`\frac{\mathrm{d}\mathbf{x}}{\mathrm{d}\boldsymbol{\lambda}}`.

Note that in the above we have used 1-based indexing as is common in mathematical notation, whereas the actual
implementation uses 0-based indexing. Additionally, :math:`\mathbf{a},\mathbf{b},\mathbf{c},\mathbf{y}` all have a batch dimension. The solver and its
:func:`~marinade.gotm.tridiagonal.TridiagonalSolve.jvp()` method support multiple r.h.s. vectors for each batch element. Variation of :math:`\mathbf{a},\mathbf{b},\mathbf{c},\mathbf{y}` across these r.h.s.
vectors is optional.
