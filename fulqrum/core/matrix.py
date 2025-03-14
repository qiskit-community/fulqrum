# Fulqrum
# Copyright (C) 2024, IBM

"""Fulqrum CSR matrix"""
import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.linalg import LinearOperator

from fulqrum.core.csr import matvec


class FulqrumCSR(csr_array, LinearOperator):

    def _matvec(self, x):
        dim = self.shape[0]
        out = np.zeros(dim, dtype=complex)
        col_vec = False
        if len(x.shape) == 2:
            col_vec = True
            x = x.view().reshape(x.shape[0])
        if self.indptr[dim] != 0:  # only do if matrix is not empty (all zeros)
            matvec(self.data, self.indices, self.indptr, x, out, dim)
        if col_vec:
            out = out.view().reshape(dim, 1)
        return out
