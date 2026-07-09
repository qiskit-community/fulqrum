# This code is a part of Fulqrum.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Spin operators for imposing spin constraints in SQD calculations.

Inspired by PySCF's ``fci.addons.fix_spin_``.

Two penalty regimes are supported, matching PySCF:

* Linear: H + lambda * (S^2 - s(s+1)).
* Quadratic: H + lambda * (S^2 - s(s+1))^2`.

The penalized operator can be built in either of the two forms used for
eigensolving in Fulqrum:

* `make_spin_penalized_operator` returns a
  `scipy.sparse.linalg.LinearOperator` (applies `S^2` twice
  for the quadratic case, so no operator squaring is needed). Both
  ``SubspaceHamiltonian`` and
  its ``to_csr_linearoperator_fast()`` CSR form work.
* `make_spin_penalized_csr` returns an explicit ``scipy.sparse`` CSR matrix.
    Avoid using it for now.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator

from ..core.fermi_operator import FermionicOperator


def s_z_fermionic_op(num_orbitals):
    n = num_orbitals
    terms = [("1", [p], 0.5) for p in range(n)]  # + 1/2 n_{pa}
    terms += [("1", [p + n], -0.5) for p in range(n)]  # - 1/2 n_{pb}
    return FermionicOperator(2 * n, terms)


def s_plus_fermionic_op(num_orbitals):
    n = num_orbitals
    return FermionicOperator(2 * n, [("+-", [p, p + n], 1.0) for p in range(n)])


def s_minus_fermionic_op(num_orbitals):
    n = num_orbitals
    return FermionicOperator(2 * n, [("+-", [p + n, p], 1.0) for p in range(n)])


def s_squared_fermionic_op(num_orbitals):
    s_z = s_z_fermionic_op(num_orbitals)
    s_plus = s_plus_fermionic_op(num_orbitals)
    s_minus = s_minus_fermionic_op(num_orbitals)
    return s_minus @ s_plus + s_z @ s_z + s_z


def s_squared_qubit_op(num_orbitals):
    """Build S^2 operator."""
    op = s_squared_fermionic_op(num_orbitals).extended_jw_transformation()
    op.set_type(1)
    return op


def assert_s2_type1(s2_qubit_op):
    r"""Guard: reject a `S^2` qubit operator that is not ``type=1``.

    Type-2 path uses a special logic. S^2 may not satisfy those. So, we set
    S^2 to be type-1 to be safer.
    """
    op_type = getattr(s2_qubit_op, "type", None)
    if op_type != 1:
        raise ValueError(
            "S^2 qubit operator must be type=1; got type="
            f"{op_type}. Build S^2 with "
            "s_squared_qubit_op(...) (type-1 by default) or call "
            "s2_qop.set_type(1)."
        )


def make_spin_penalized_fermionic_op(
    hamiltonian,
    num_orbitals,
    num_a,
    num_b,
    target_s,
    lam=0.2,
):
    """Build a spin-penalized Hamiltonian as a single `FermionicOperator`.

    Parameters:
        hamiltonian (FermionicOperator): Hamiltonian.
        num_orbitals (int): Number of spatial orbitals ``N``.
        num_a (int): Number of spin-up (alpha) electrons.
        num_b (int): Number of spin-down (beta) electrons.
        target_s (float): Target total spin quantum number ``s``.
        lam (float): Penalty weight `lambda`. Defaults to ``0.2``.

    Returns:
        FermionicOperator: ``hamiltonian + lam * P(S^2)``.
    """
    num_qubits = 2 * num_orbitals
    s2 = s_squared_fermionic_op(num_orbitals)
    ss = target_s * (target_s + 1.0)
    # (S^2 - ss I) as a fermionic operator (identity is the empty-string term).
    dev = s2 + FermionicOperator(num_qubits, [("", [], -ss)])
    penalty = dev @ dev if use_quadratic_penalty(num_a, num_b, target_s) else dev
    return hamiltonian + lam * penalty


def use_quadratic_penalty(num_a, num_b, target_s):
    r"""Decide whether the quadratic spin penalty is required.

    Mirrors PySCF's ``fix_spin`` logic.

    Parameters:
        num_a (int): Number of spin-up (alpha) electrons.
        num_b (int): Number of spin-down (beta) electrons.
        target_s (float): Target total spin quantum number ``s``.

    Returns:
        bool: ``True`` if the quadratic penalty is required.
    """
    sz = abs(num_a - num_b) * 0.5
    ss = target_s * (target_s + 1.0)
    # PySCF uses the linear form when ss < sz*(sz+1) + 0.1 (target is the lowest
    # spin of the sector); otherwise the quadratic form.
    return ss >= sz * (sz + 1.0) + 0.1


def make_spin_penalized_operator(
    h_operator,
    s2_operator,
    num_a,
    num_b,
    target_s,
    lam=0.2,
):
    r"""Compose a spin-penalized Hamiltonian as a ``LinearOperator``.

    Both ``h_operator`` and ``s2_operator`` must be projected onto the *same*
    subspace and expose a ``matvec`` and a matching ``shape``.

    Parameters:
        h_operator: Projected molecular Hamiltonian (has ``matvec`` / ``shape``).
        s2_operator: Projected `S^2` operator on the same subspace.
        num_a (int): Number of spin-up (alpha) electrons.
        num_b (int): Number of spin-down (beta) electrons.
        target_s (float): Target total spin quantum number ``s`` (e.g. ``0`` for a
            singlet, ``0.5`` doublet, ``1`` triplet).
        lam (float): Penalty weight. Defaults to ``0.2`` (the
            value used in arXiv:2411.04827; PySCF's ``fix_spin_`` uses 0.1-0.2).
            A higher value enforces the constraint more strongly.

    Returns:
        scipy.sparse.linalg.LinearOperator: The spin-penalized operator.
    """
    if h_operator.shape != s2_operator.shape:
        raise ValueError(
            "h_operator and s2_operator must act on the same subspace; got shapes "
            f"{h_operator.shape} and {s2_operator.shape}."
        )

    ss = target_s * (target_s + 1.0)
    quadratic = use_quadratic_penalty(num_a, num_b, target_s)
    dim = h_operator.shape[0]
    # The penalty mixes a complex S^2 action into a possibly-real H; keep the
    # composite complex unless both inputs are real.
    is_real = getattr(h_operator, "dtype", np.dtype(complex)) == np.dtype(
        float
    ) and getattr(s2_operator, "dtype", np.dtype(complex)) == np.dtype(float)
    dtype = np.dtype(float) if is_real else np.dtype(complex)

    def matvec(x):
        x = np.asarray(x).reshape(dim)
        hv = h_operator.matvec(x)
        dev = s2_operator.matvec(x) - ss * x  # (S^2 - ss) v
        if quadratic:
            penalty = s2_operator.matvec(dev) - ss * dev  # (S^2 - ss)^2 v
        else:
            penalty = dev
        return np.asarray(hv).reshape(dim) + lam * penalty

    return LinearOperator((dim, dim), matvec=matvec, dtype=dtype)


def make_spin_penalized_csr(
    h_csr,
    s2_csr,
    num_a,
    num_b,
    target_s,
    lam=0.2,
):
    """Avoid using it for now"""
    # Accept either a raw scipy sparse matrix or a Fulqrum CSRLinearOperator.
    h_mat = h_csr.matrix if hasattr(h_csr, "matrix") else h_csr
    s2_mat = s2_csr.matrix if hasattr(s2_csr, "matrix") else s2_csr
    if h_mat.shape != s2_mat.shape:
        raise ValueError(
            "h_csr and s2_csr must act on the same subspace; got shapes "
            f"{h_mat.shape} and {s2_mat.shape}."
        )

    ss = target_s * (target_s + 1.0)
    quadratic = use_quadratic_penalty(num_a, num_b, target_s)

    identity = sp.identity(h_mat.shape[0], dtype=s2_mat.dtype, format="csr")
    dev = sp.csr_matrix(s2_mat) - ss * identity  # (S^2 - ss I)
    penalty = dev @ dev if quadratic else dev

    return (sp.csr_matrix(h_mat) + lam * penalty).tocsr()
