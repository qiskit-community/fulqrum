# Fulqrum
# Copyright (C) 2024, IBM
# cython: c_string_type=unicode, c_string_encoding=UTF-8

cimport cython

include "fulqrum/core/includes/csr_header.pxi"


def matvec(complex[::1] data, fused_int[::1] indices, fused_int[::1] indptr,
            complex[::1] x, complex[::1] out, size_t dim):
    """Dispatch matvec based on indices type
    """
    if fused_int == int:    
        csr_matvec[int](&data[0], &indices[0], &indptr[0], &x[0], &out[0], dim)
    else:
        csr_matvec[int64](&data[0], &indices[0], &indptr[0], &x[0], &out[0], dim)
