# Fulqrum
# Copyright (C) 2024, IBM
# cython: c_string_type=unicode, c_string_encoding=UTF-8

cimport cython
from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.string cimport memcpy
from libc.math cimport floor, sqrt

from fulqrum.core.qubit_operator cimport QubitOperator
from fulqrum.core.subspace cimport Subspace
from fulqrum.exceptions import FulqrumError
#from fulqrum.core.csr cimport csr_matrix_builder

from cython.parallel cimport prange, parallel
import time
import numpy as np
import scipy.sparse as sp
import psutil
cimport numpy as np
np.import_array()

include "includes/base_header.pxi"
include "includes/elements_header.pxi"
include "includes/diag_header.pxi"
include "includes/matvec_header.pxi"
include "includes/matvec2_header.pxi"
include "includes/csr_header.pxi"
include "includes/csr_utils_header.pxi"
include "includes/csr2_header.pxi"
include "includes/grouping_header.pxi"


cdef class FulqrumSpMV():
    def __cinit__(self, QubitOperator diag_hamiltonian,
                  QubitOperator hamiltonian, Subspace subspace,
                  size_t[::1]& group_ptrs, size_t[::1]& group_ladder_ptrs):
        cdef size_t kk
        self.diag_oper = diag_hamiltonian.oper
        self.oper = hamiltonian.oper
        self.is_real = diag_hamiltonian.is_real() and hamiltonian.is_real()
        self.subspace = subspace
        self.bin_width = self.subspace.subspace.bin_width
        self.width = self.oper.width
        self.subspace_dim = self.subspace.subspace.bitstrings.size()
        self.num_terms = self.oper.terms.size()
        self.num_diag_terms = self.diag_oper.terms.size()
        self.bin_ranges = &self.subspace.subspace.bin_ranges[0]
        self.group_ptrs = group_ptrs
        self.group_ladder_ptrs = group_ladder_ptrs
        self.num_groups = group_ptrs.shape[0] - 1
        if group_ptrs.shape[0] > 1:
            set_group_offdiag_indices(self.oper.terms, self.group_offdiag_inds,
                                      &self.group_ptrs[0], self.num_groups)
        
        if self.oper.type == 2:
            self.group_rowint_length = hamiltonian.group_rowint_length()
            self.ladder_offset = 2**self.oper.ladder_width

        if self.diag_oper.terms.size() > 0:
            self.has_nonzero_diag = 1
             # Init diagonal memoryview to None because
             # we only build it when needed
            self.complex_diag_vec = None
        else:
            self.has_nonzero_diag = 0
            # We have to init something here otherwise
            # grabbing a pointer to the data is going to complain
            self.complex_diag_vec = np.empty(1, dtype=complex)

    def __repr__(self):
        out = f"<FulqrumSpMV(width={self.width}, "
        out += f"num_op_terms={self.num_terms+self.num_diag_terms}({self.num_terms}/{self.num_diag_terms}), "
        out += f"subspace_dim={self.subspace_dim}, "
        out += f"is_real={self.is_real}>"
        return out
    
    def is_real(self):
        """Flag indicating the problem is real (symmetry) or complex (Hermitian)

        Returns:
            int: 1 if real, else 0 for complex
        """
        return self.is_real

    @cython.boundscheck(False)
    cdef void compute_diag_vector(self):
        self.complex_diag_vec = np.empty(self.subspace_dim, dtype=complex)
        compute_diag_vector(self.subspace.subspace.bitstrings,
                            &self.complex_diag_vec[0],
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
        if self.complex_diag_vec is None and self.has_nonzero_diag:
            self.compute_diag_vector()
        return np.asarray(self.complex_diag_vec)


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
        if self.complex_diag_vec.shape[0] == 0 and self.has_nonzero_diag:
            self.compute_diag_vector()
        cdef double complex[::1] out = np.zeros(x.shape[0], dtype=complex)
        if self.oper.type == 2:
            omp_matvec2(self.oper.terms,
                    self.subspace.subspace.bitstrings,
                    &self.complex_diag_vec[0],
                    self.width,
                    self.subspace_dim,
                    self.has_nonzero_diag,
                    self.bin_width,
                    self.bin_ranges,
                    &self.group_ptrs[0],
                    &self.group_ladder_ptrs[0],
                    &self.group_rowint_length[0],
                    self.group_offdiag_inds,
                    self.num_groups,
                    self.ladder_offset,
                    &x[0],
                    &out[0])
        else:
            omp_matvec(self.oper.terms,
                    self.subspace.subspace.bitstrings,
                    &self.complex_diag_vec[0],
                    self.width,
                    self.subspace_dim,
                    self.has_nonzero_diag,
                    self.bin_width,
                    self.bin_ranges,
                    &self.group_ptrs[0],
                    self.group_offdiag_inds,
                    self.num_groups,
                    &x[0],
                    &out[0])
        return np.asarray(out)

    
    def to_csr_array(self, int verbose=0):
        """Convert subspace Hamiltonian to a SciPy CSR array

        Parameters:
            verbose (int): Turn on or off verbose mode, default=0.

        Returns:
            csr_array: Sparse representation of subspace Hamiltonian
        """
        cdef size_t max_int = np.iinfo(np.int32).max
        cdef size_t num_terms = self.oper.terms.size()
        
        cdef int[::1] indptr32
        cdef int[::1] indices32
        cdef int64[::1] indptr64
        cdef int64[::1] indices64
        cdef double complex[::1] data = np.zeros(1, dtype=complex)

        indptr64 = np.zeros(self.subspace_dim+1, dtype=np.int64)
        indices64 = np.zeros(1, dtype=np.int64)


        if self.complex_diag_vec.shape[0] == 0 and self.has_nonzero_diag:
            self.compute_diag_vector()
        cdef double start, stop
        cdef int compute_values;
        cdef int64 total_bytes;
        cdef int int_64 = 1 # always start with 64bit ints
        for compute_values in range(2):
            start = time.perf_counter()
            if compute_values:
                # matrix is empty
                if indptr64[self.subspace_dim] == 0:
                        return sp.csr_array((self.subspace_dim, self.subspace_dim), dtype=complex)

                 # if num_elem > int32 or subspace_dim + 1 > int32
                if (indptr64[self.subspace_dim] < max_int) and ((self.subspace_dim + 1) < max_int):
                    int_64 = 0
                
                # check if matrix will fit into memory
                if int_64:
                    # indptr + indices + data sizes
                    total_bytes = (self.subspace_dim + 1) * 8  + indptr64[self.subspace_dim] * 8 + indptr64[self.subspace_dim] * 16
                else:
                    total_bytes = (self.subspace_dim + 1) * 4  + indptr64[self.subspace_dim] * 4 + indptr64[self.subspace_dim] * 16
                if psutil.virtual_memory().available < total_bytes:
                    raise FulqrumError(f"Sparse matrix of size {total_bytes/(1024**3)}Gb does not fit within available memory.")

                if int_64:
                    indices64 = np.zeros(indptr64[self.subspace_dim], dtype=np.int64)
                    data = np.zeros(indptr64[self.subspace_dim], dtype=complex)
                else:
                    indptr32 = np.asarray(indptr64, dtype=np.int32)
                    indptr64 = np.zeros(1, dtype=np.int64)
                    indices32 = np.zeros(indptr32[self.subspace_dim], dtype=np.int32)
                    data = np.zeros(indptr32[self.subspace_dim], dtype=complex)
                
            if int_64:
                if self.oper.type == 2:
                    csr_matrix_builder2[int64](&self.oper.terms[0],
                                        self.subspace.subspace.bitstrings,
                                        &self.complex_diag_vec[0],
                                        self.width,
                                        self.subspace_dim,
                                        self.has_nonzero_diag,
                                        self.bin_width,
                                        self.bin_ranges,
                                        &self.group_ptrs[0],
                                        &self.group_ladder_ptrs[0],
                                        &self.group_rowint_length[0],
                                        self.group_offdiag_inds,
                                        self.num_groups,
                                        self.ladder_offset,
                                        &indptr64[0],
                                        &indices64[0],
                                        &data[0],
                                        compute_values)
                else:
                    csr_matrix_builder[int64](&self.oper.terms[0],
                                        self.subspace.subspace.bitstrings,
                                        &self.complex_diag_vec[0],
                                        self.width,
                                        self.subspace_dim,
                                        self.has_nonzero_diag,
                                        self.bin_width,
                                        self.bin_ranges,
                                        &self.group_ptrs[0],
                                        self.group_offdiag_inds,
                                        self.num_groups,
                                        &indptr64[0],
                                        &indices64[0],
                                        &data[0],
                                        compute_values)
            else:
                if self.oper.type == 2:
                    csr_matrix_builder2[int](&self.oper.terms[0],
                                        self.subspace.subspace.bitstrings,
                                        &self.complex_diag_vec[0],
                                        self.width,
                                        self.subspace_dim,
                                        self.has_nonzero_diag,
                                        self.bin_width,
                                        self.bin_ranges,
                                        &self.group_ptrs[0],
                                        &self.group_ladder_ptrs[0],
                                        &self.group_rowint_length[0],
                                        self.group_offdiag_inds,
                                        self.num_groups,
                                        self.ladder_offset,
                                        &indptr32[0],
                                        &indices32[0],
                                        &data[0],
                                        compute_values)
                else:
                    csr_matrix_builder[int](&self.oper.terms[0],
                                        self.subspace.subspace.bitstrings,
                                        &self.complex_diag_vec[0],
                                        self.width,
                                        self.subspace_dim,
                                        self.has_nonzero_diag,
                                        self.bin_width,
                                        self.bin_ranges,
                                        &self.group_ptrs[0],
                                        self.group_offdiag_inds,
                                        self.num_groups,
                                        &indptr32[0],
                                        &indices32[0],
                                        &data[0],
                                        compute_values)
            stop = time.perf_counter()
            if verbose:
                if not compute_values:
                    print('CSR structure time', round(stop-start, 3))
                else:
                    print('CSR fill time', round(stop-start, 3))
        if int_64:
            mat = sp.csr_array((data, indices64, indptr64), 
                            shape=(self.subspace_dim,)*2, dtype=complex)
        else:
            mat = sp.csr_array((data, indices32, indptr32), 
                            shape=(self.subspace_dim,)*2, dtype=complex)
        start = time.perf_counter()
        quicksort_indices(mat.indices, mat.indptr, mat.data)
        stop = time.perf_counter()
        if verbose:
            print('CSR indices sort time', round(stop-start, 3))
        return mat


@cython.boundscheck(False)
def quicksort_indices(int32_or_int64[::1] indices,
                      int32_or_int64[::1] indptr,
                      double_or_complex[::1] data):
    cdef int32_or_int64 kk, nrows = indptr.shape[0]-1
    cdef int32_or_int64 start, stop
    for kk in prange(nrows, nogil=True):
        start = indptr[kk]
        stop = indptr[kk+1] - 1
        quicksort_indices_data(&indices[0], &data[0], start, stop)
