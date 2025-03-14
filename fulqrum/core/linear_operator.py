# Fulqrum
# Copyright (C) 2024, IBM

"""Fulqrum linearoperator module"""
import numpy as np
from scipy.sparse.linalg import LinearOperator

from .spmv import FulqrumSpMV


class SubspaceHamiltonian(LinearOperator):
    """Encapsulates the details of a subspace Hamiltonian problem
    and can be passed to SciPy eigensolvers for matrix-free
    evaluation.
    """

    # The subclass will complain if this is not here
    _matvec = None

    def __init__(self, hamiltonian, subspace):
        diag_H, off_H = hamiltonian.split_diagonal()
        if not off_H.sorted:
            off_H.offdiag_term_grouping()
        self.spmv = FulqrumSpMV(diag_H, off_H, subspace)
        self._matvec = self.matvec
        self.shape = (len(subspace),) * 2
        self.dtype = np.dtype(complex)

    def interpret_vector(self, vec, sort=0):
        return self.spmv.subspace.interpret_vector(vec, sort)

    def __repr__(self):
        out = f"<SubspaceHamiltonian(width={self.spmv.width}, "
        out += f"num_op_terms={self.spmv.num_diag_terms+self.spmv.num_terms}({self.spmv.num_diag_terms}/{self.spmv.num_terms}), "
        out += f"subspace_dim={self.spmv.subspace_dim}>"
        return out

    def matvec(self, x):
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

    def to_csr(self):
        return self.spmv.to_csr()
