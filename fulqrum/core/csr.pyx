# Fulqrum
# Copyright (C) 2024, IBM
# cython: c_string_type=unicode, c_string_encoding=UTF-8
cimport cython

include "includes/csr_header.pxi"
include "includes/types.pxi"



def csr_matvec(int32_or_int64[::1] indptr, int32_or_int64[::1] indices, double_or_complex[::1] data, 
               double_or_complex[::1] vec, double_or_complex[::1] out, size_t dim):
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
