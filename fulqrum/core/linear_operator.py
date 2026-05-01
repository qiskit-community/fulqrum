# This code is a part of Fulqrum.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

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
        """A SciPy `LinearOperator` that represents a Hamiltonian restricted to the given
        subspace.
        """
        self.diag_H, self.off_H = hamiltonian.split_diagonal()
        # if there are no off-diagonal terms then we pass a dummy empty array of len=1
        self.off_H.offdiag_term_grouping()
        self.group_ptrs = np.zeros(1, dtype=np.uintp)
        self.group_ladder_ptrs = np.zeros(1, dtype=np.uintp)

        if self.off_H.num_terms:
            self.group_ptrs = self.off_H.group_ptrs()
        if self.off_H.type == 2:
            if self.off_H.num_terms:
                self.off_H.group_term_sort_by_ladder_int(4)
                self.group_ladder_ptrs = self.off_H.group_ladder_bin_starts()

        self.spmv = FulqrumSpMV(
            self.diag_H, self.off_H, subspace, self.group_ptrs, self.group_ladder_ptrs
        )
        self._matvec = self.matvec
        self.shape = (len(subspace),) * 2
        self.dtype = np.dtype(float) if self.spmv.is_real else np.dtype(complex)

    @property
    def num_groups(self):
        """Number of off-diagonal groupings

        Returns:
            int : Number of groups in operator
        """
        return self.off_H.num_groups

    def diagonal_vector(self, verbose=False, disable_fast_mode=False):
        """Return diagonal vector of Hamiltonian in subspace

        Parameters:
            verbose (bool): optional, verbose output, default=False
            disable_fast_mode (bool): optional, disable fast computation for type=2 Hamiltonians, default=False
        Returns:
            ndarray: Complex vector for diagonal of Hamiltonian
        """
        return self.spmv.diagonal_vector(verbose, disable_fast_mode)

    def minimum_diagonal_energy(self):
        """Return the minimum diagonal energy

        Returns:
            double: Lowest energy value
        """
        return self.spmv.minimum_diagonal_energy()

    def interpret_vector(self, vec, atol=1e-14, sort=False, renormalize=True):
        """Convert solution vector into dict of counts and real or complex amplitudes

        Parameters:
            vec (ndarray): Complex or real solution vector
            atol (double): Absolute tolerance for truncation, default=1e-14
            sort (bool): Sort output dict by integer representation, default=False.
            renormalize (bool): Renormalize values such that probabilities sum to one, default = True

        Returns:
            dict: Dictionary with bit-string keys and complex values

        Note:
            Truncation can be disabled by calling `atol=0`
        """
        if len(vec.shape) == 2:
            vec = vec.view().reshape(vec.shape[0])
        return self.spmv.subspace.interpret_vector(vec, atol, sort, renormalize)

    def get_n_th_bitstring(self, n):
        """Return n-th bitstring in the SubspaceHamiltonian

        Parameters:
            n (int): Index of the expected bitstring.

        Returns:
            str: N-th bitstring in the subspace.

        Note:
            Both Python dictionaries and `emhash8::HashMap` retain insertion order.

        Raises:
            ValueError: If ``n`` is >= to the number of bit-strings in the subspace.
        """
        if n >= self.spmv.subspace.size():
            raise ValueError(
                f"Value of n ({n}) >= subspace size {self.spmv.subspace.size()}"
            )

        return self.spmv.subspace.get_n_th_bitstring(n)

    def __repr__(self):
        out = f"<SubspaceHamiltonian(width={self.spmv.width}, "
        out += f"num_op_terms={self.spmv.num_diag_terms + self.spmv.num_terms}({self.spmv.num_diag_terms}/{self.spmv.num_terms}), "
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

    def to_csr_linearoperator(self, verbose=False):
        """Convert subspace Hamiltonian to a LinearOperator wrapping a CSR matrix

        Parameters:
            verbose (bool): Turn on verbose mode, default=False.

        Returns:
            CSRLinearOperator: LinearOperator wrapping a CSR matrix.
        """
        M = self.spmv.to_csr_array(verbose=verbose)
        return CSRLinearOperator(M, self.spmv.is_real)

    def to_csr_linearoperator_fast(self, verbose=False):
        """Convert subspace Hamiltonian to a CSR LinearOperator faster but with a copy

        Parameters:
            verbose (bool): Turn on verbose mode, default=False.
        """
        M = self.spmv.to_csrlike(verbose).to_csr_array(verbose)
        return CSRLinearOperator(M, self.spmv.is_real)

    def _to_linearoperator(self, verbose=False):
        """Convert subspace Hamiltonian to a CSR-like format LinearOperator

        This saves a matrix-traversal at the expense of a non-standard data type

        Parameters:
            verbose (bool): Turn on verbose mode, default=False.
        """
        out = self.spmv.to_csrlike(verbose)
        return out


class CSRLinearOperator(LinearOperator):
    _matvec = None

    def __init__(self, matrix, is_real=0):
        """A SciPy `LinearOperator` wrapper for a CSR array.  Allows for parallel
        computation of the sparse-matrix vector product needed for eigensolving.
        """
        self.matrix = matrix
        self.is_real = is_real
        super().__init__(shape=matrix.shape, dtype=float if self.is_real else complex)

    @property
    def nnz(self):
        """Number of nonzero elements in the underlying CSR matrix

        Returns:
            int
        """
        return self.matrix.nnz

    @property
    def memory_size(self):
        """An estimate of the raw memory size of the underlying
        CSR array in bytes

        Returns:
            int: Memory size in bytes
        """
        nnz = self.nnz
        inds_size = 4
        data_size = 8
        if self.matrix.indices.dtype == np.int64:
            inds_size = 8
        if self.matrix.data.dtype in [complex, np.complex128]:
            data_size = 16
        return inds_size * (self.shape[0] + 1) + nnz * (inds_size + data_size)

    def matvec(self, x):
        col_vec = False
        if len(x.shape) == 2:
            col_vec = True
            x = x.view().reshape(
                x.shape[0],
            )
        out = np.zeros_like(x, dtype=float if self.is_real else complex)
        csr_matvec(
            self.matrix.indptr,
            self.matrix.indices,
            self.matrix.data,
            x,
            out,
            x.shape[0],
        )
        if col_vec:
            out = out.view().reshape(x.shape[0], 1)
        return out


class CSRLikeLinearOperator(LinearOperator):
    _matvec = None

    def __init__(self, csrlike):
        """A SciPy `LinearOperator` wrapper for a CSR like object consisting of a vector of objects,
        each comprised of two vectors for column indices and data .  Can be quickly converted to a CSR array,
        and Allows for parallel computation of the sparse-matrix vector product needed for eigensolving,
        but is less efficient than canonical CSR format.
        """
        self.csrlike = csrlike
        self.is_real = csrlike.is_real
        super().__init__(shape=csrlike.shape, dtype=float if self.is_real else complex)

    def to_csr_array(self, verbose=False):
        return self.csrlike.to_csr_array(verbose)

    @property
    def nnz(self):
        """Number of nonzero elements in the underlying data structure

        Returns:
            int
        """
        return self.csrlike.nnz

    @property
    def memory_size(self):
        """An estimate of the raw memory size of the underlying
        date structure in bytes

        Returns:
            int: Memory size in bytes
        """
        nnz = self.nnz
        inds_size = 4
        data_size = 8
        if "64" in self.csrlike.type_string:
            inds_size = 8
        if "z" in self.csrlike.type_string:
            data_size = 16
        return inds_size * (self.shape[0] + 1) + nnz * (inds_size + data_size)

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
