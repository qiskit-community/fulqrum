# Fulqrum
# Copyright (C) 2024, IBM

"""Fulqrum linearoperator module"""
import numpy as np
from scipy.sparse.linalg import LinearOperator

from .spmv import FulqrumSpMV
from .csr import csr_matvec


class SubspaceHamiltonian(LinearOperator):
    """Encapsulates the details of a subspace Hamiltonian problem
    and can be passed to SciPy eigensolvers for matrix-free
    evaluation.
    """

    # The subclass will complain if this is not here
    _matvec = None

    def __init__(self, hamiltonian, subspace):
        diag_H, off_H = hamiltonian.split_diagonal()
        # if there are no off-diagonal terms then we pass a dummy empty array of len=1
        self.group_ptrs = (
            off_H.group_ptrs() if off_H.num_terms else np.zeros(1, dtype=np.uintp)
        )
        self.spmv = FulqrumSpMV(
            diag_H,
            off_H,
            subspace,
            self.group_ptrs,
        )
        self._matvec = self.matvec
        self.shape = (len(subspace),) * 2
        self.dtype = np.dtype(complex)

    def diagonal_vector(self):
        """Return diagonal vector of Hamiltonian in subspace

        Returns:
            ndarray: Complex vector for diagonal of Hamiltonian
        """
        return self.spmv.diagonal_vector()

    def interpret_vector(self, vec, atol=1e-14, sort=0):
        """Convert solution vector into dict of counts and complex amplitudes

        Parameters:
            vec (ndarray): Complex solution vector
            atol (double): Absolute tolerance for truncation, default=1e-14
            sort (int): Sort output dict by integer representation.

        Returns:
            dict: Dictionary with bit-string keys and complex values

        Notes:
            Truncation can be disabled by calling `atol=-1`
        """
        if len(vec.shape) == 2:
            vec = vec.view().reshape(vec.shape[0])
        return self.spmv.subspace.interpret_vector(vec, atol, sort)

    def __repr__(self):
        out = f"<SubspaceHamiltonian(width={self.spmv.width}, "
        out += f"num_op_terms={self.spmv.num_diag_terms+self.spmv.num_terms}({self.spmv.num_diag_terms}/{self.spmv.num_terms}), "
        out += f"subspace_dim={self.spmv.subspace_dim}>"
        return out

    def matvec(self, x):
        """Matrix-free implimentation of SpMV for subspace Hamiltonian

        Parameters:
            x (ndarray): Complex-valued input array

        Returns:
            ndarray: Complex output vector after SpMV on input vector
        """
        col_vec = False
        if len(x.shape) == 2:
            col_vec = True
            x = x.view().reshape(
                x.shape[0],
            )
        out = self.spmv.matvec(x)
        if col_vec:
            out = out.view().reshape(x.shape[0], 1)
        return out

    def to_csr_array(self):
        """Convert subspace Hamiltonian to a SciPy CSR array

        Returns:
            csr_array: Sparse representation of subspace Hamiltonian
        """
        return self.spmv.to_csr_array()

    def to_csr_linearoperator(self):
        """Convert subspace Hamiltonian to a LinearOperator wrapping a CSR matrix

        Returns:
            CSRLinearOperator: LinearOperator wrapping a CSR matrix.
        """
        M = self.spmv.to_csr_array()
        return CSRLinearOperator(M)


class CSRLinearOperator(LinearOperator):
    _matvec = None

    def __init__(self, mat):
        self.mat = mat
        super().__init__(shape=mat.shape, dtype=complex)

    def matvec(self, x):
        col_vec = False
        if len(x.shape) == 2:
            col_vec = True
            x = x.view().reshape(
                x.shape[0],
            )
        out = np.zeros_like(x, dtype=complex)
        csr_matvec(self.mat.indptr, self.mat.indices, self.mat.data, x, out, x.shape[0])
        if col_vec:
            out = out.view().reshape(x.shape[0], 1)
        return out
