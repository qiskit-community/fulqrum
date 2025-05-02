# Fulqrum
# Copyright (C) 2024, IBM
# cython: c_string_type=unicode, c_string_encoding=UTF-8
cimport cython

include "includes/csr_header.pxi"

ctypedef long long int64

ctypedef fused fused_type:
    int
    int64


def csr_matvec(fused_type[::1] indptr, fused_type[::1] indices, complex[::1] data, 
               complex[::1] vec, complex[::1] out, size_t dim):
    """Perform SpMV using a CSR matrix

    Parameters:
        indptr (ndarray): int or int64 array of row pointers
        indices (ndarray): int or int64 column indices of each nonzero element
        data (ndarray): double complex data of matrix
        vec (ndaray): double complex input vector
        out (ndaray): double complex output vector
        dim (size_t): dimension of system
    """
    csr_spmv(&indptr[0], &indices[0], &data[0], &vec[0], &out[0], dim)
