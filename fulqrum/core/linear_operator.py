# Fulqrum
# Copyright (C) 2024, IBM

"""Fulqrum linearoperator module"""
import numpy as np
from scipy.sparse.linalg import LinearOperator

from .spmv import FulqrumSpMV
from .csr import csr_matvec
from fulqrum.core.csrlike import CSRLike


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
        off_H.offdiag_term_grouping()
        self.group_ptrs = np.zeros(1, dtype=np.uintp)
        self.group_ladder_ptrs = np.zeros(1, dtype=np.uintp)

        if off_H.num_terms:
            self.group_ptrs = off_H.group_ptrs()
        if off_H.type == 2:
            if off_H.num_terms:
                off_H.group_term_sort_by_ladder_int(4)
                self.group_ladder_ptrs = off_H.group_ladder_bin_starts()

        self.spmv = FulqrumSpMV(
            diag_H, off_H, subspace, self.group_ptrs, self.group_ladder_ptrs
        )
        self._matvec = self.matvec
        self.shape = (len(subspace),) * 2
        self.dtype = np.dtype(float) if self.spmv.is_real else np.dtype(complex)

    def diagonal_vector(self):
        """Return diagonal vector of Hamiltonian in subspace

        Returns:
            ndarray: Complex vector for diagonal of Hamiltonian
        """
        return self.spmv.diagonal_vector()

    def interpret_vector(self, vec, atol=1e-12, sort=0):
        """Convert solution vector into dict of counts and complex amplitudes

        Parameters:
            vec (ndarray): Complex solution vector
            atol (double): Absolute tolerance for truncation, default=1e-12
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
        """Matrix-free implementation of SpMV for subspace Hamiltonian

        Parameters:
            x (ndarray): Input array

        Returns:
            ndarray: Output vector after SpMV on input vector
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

    def to_csr_array(self, verbose=False):
        """Convert subspace Hamiltonian to a SciPy CSR array

        Parameters:
            verbose (bool): Turn on verbose mode, default=False.

        Returns:
            csr_array: Sparse representation of subspace Hamiltonian
        """
        return self.spmv.to_csr_array(verbose=verbose)

    def to_csr_linearoperator(self, verbose=False):
        """Convert subspace Hamiltonian to a LinearOperator wrapping a CSR matrix

        Parameters:
            verbose (bool): Turn on verbose mode, default=False.

        Returns:
            CSRLinearOperator: LinearOperator wrapping a CSR matrix.
        """
        M = self.spmv.to_csr_array(verbose=verbose)
        return CSRLinearOperator(M, self.spmv.is_real)

    def to_csrlike_linearoperator(self, verbose=False):
        """Convert subspace Hamiltonian to a CSR-like format LinearOperator

        This saves a matrix-traversal at the expense of a non-standard data type

        Parameters:
            verbose (bool): Turn on verbose mode, default=False.
        """
        out = self.spmv.to_csrlike()
        return out


class CSRLinearOperator(LinearOperator):
    _matvec = None

    def __init__(self, mat, is_real=0):
        self.mat = mat
        self.is_real = is_real
        super().__init__(shape=mat.shape, dtype=float if self.is_real else complex)

    def matvec(self, x):
        col_vec = False
        if len(x.shape) == 2:
            col_vec = True
            x = x.view().reshape(
                x.shape[0],
            )
        out = np.zeros_like(x, dtype=float if self.is_real else complex)
        csr_matvec(self.mat.indptr, self.mat.indices, self.mat.data, x, out, x.shape[0])
        if col_vec:
            out = out.view().reshape(x.shape[0], 1)
        return out


class CSRLikeLinearOperator(LinearOperator):
    _matvec = None

    def __init__(self, csrlike):
        self.csrlike = csrlike
        self.is_real = csrlike.is_real
        super().__init__(shape=csrlike.shape, dtype=float if self.is_real else complex)

    def to_csr_array(self):
        return self.csrlike.to_csr_array()
    
    
    def matvec(self, x):
        """Matrix-free implementation of SpMV for subspace Hamiltonian

        Parameters:
            x (ndarray): Input array

        Returns:
            ndarray: Output vector after SpMV on input vector
        """
        col_vec = False
        if len(x.shape) == 2:
            col_vec = True
            x = x.view().reshape(
                x.shape[0],
            )
        out = self.csrlike.matvec(x)
        if col_vec:
            out = out.view().reshape(x.shape[0], 1)
        return out
