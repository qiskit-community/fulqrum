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
