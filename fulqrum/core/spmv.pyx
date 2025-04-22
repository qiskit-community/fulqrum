# Fulqrum
# Copyright (C) 2024, IBM
# cython: c_string_type=unicode, c_string_encoding=UTF-8

cimport cython
from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.string cimport memcpy

from fulqrum.core.qubit_operator cimport QubitOperator
from fulqrum.core.subspace cimport Subspace

from cython.parallel cimport prange, parallel

import numpy as np
cimport numpy as np
np.import_array()

include "includes/base_header.pxi"
include "includes/elements_header.pxi"
include "includes/bitstrings_header.pxi"
include "includes/diag_header.pxi"
include "includes/matvec_header.pxi"


cdef class FulqrumSpMV():
    def __cinit__(self, QubitOperator diag_hamiltonian,
                  QubitOperator hamiltonian, Subspace subspace, size_t[::1] group_ptrs):
        cdef size_t kk
        self.diag_oper = diag_hamiltonian.oper
        self.oper = hamiltonian.oper
        self.subspace = subspace
        self.bin_width = self.subspace.subspace.bin_width
        self.width = self.oper.width
        self.subspace_dim = self.subspace.subspace.bitstrings.size() // self.width
        self.num_terms = self.oper.terms.size()
        self.num_diag_terms = self.diag_oper.terms.size()
        self.bin_ranges = &self.subspace.subspace.bin_ranges[0]
        self.group_ptrs = &group_ptrs[0]
        self.num_groups = group_ptrs.shape[0] - 1
        if self.diag_oper.terms.size() > 0:
            self.has_nonzero_diag = 1
             # Init diagonal memoryview to None because
             # we only build it when needed
            self.diag_vec = None
        else:
            self.has_nonzero_diag = 0
            # We have to init something here otherwise
            # grabbing a pointer to the data is going to complain
            self.diag_vec = np.empty(1, dtype=complex)

    def __repr__(self):
        out = f"<FulqrumSpMV(width={self.width}, "
        out += f"num_op_terms={self.num_terms+self.num_diag_terms}({self.num_terms}/{self.num_diag_terms}), "
        out += f"subspace_dim={self.subspace_dim}>"
        return out

    @cython.boundscheck(False)
    cdef void compute_diag_vector(self):
        cdef const unsigned char * data = &self.subspace.subspace.bitstrings.data()[0]
        self.diag_vec = np.empty(self.subspace_dim, dtype=complex)
        compute_diag_vector(data,
                            &self.diag_vec[0],
                            self.diag_oper,
                            self.width,
                            self.subspace_dim)

    def diagonal_vector(self):
        """Diagonal vector of subspace Hamitlonian

        Returns:
            ndarray: Aray of complex numbers representating diagonal
        """
        if not self.has_nonzero_diag:
            return np.zeros(self.subspace_dim, dtype=complex)
        if self.diag_vec is None and self.has_nonzero_diag:
            self.compute_diag_vector()
        return np.asarray(self.diag_vec)


    def matvec(self, const double complex[::1] x):
        """Matrix-free implimentation of SpMV for subspace Hamiltonian

        Parameters:
            x (ndarray): Complex-valued input array

        Returns:
            ndarray: Complex output vector after SpMV on input vector
        """
        if <size_t>x.shape[0] != self.subspace_dim:
            raise Exception('Incorrect length of input vector.')
        # generate diagonal vector if we have not done so already
        if self.diag_vec.shape[0] == 0 and self.has_nonzero_diag:
            self.compute_diag_vector()
        cdef double complex[::1] out = np.zeros(x.shape[0], dtype=complex)
        omp_matvec(self.oper,
                   self.subspace.subspace.bitstrings,
                   &self.diag_vec[0],
                   self.width,
                   self.subspace_dim,
                   self.has_nonzero_diag,
                   self.bin_width,
                   self.bin_ranges,
                   self.group_ptrs,
                   self.num_groups,
                   &x[0],
                   &out[0])
        return np.asarray(out)
