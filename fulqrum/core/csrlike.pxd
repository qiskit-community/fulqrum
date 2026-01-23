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
include "includes/csrlike_header.pxi"
import numpy as np


cdef class CSRLike:
    cdef size_t num_rows
    cdef size_t _nnz
    cdef public unsigned int is_real
    cdef public unsigned int is_int64
    cdef unsigned int data_type
    cdef RowData_Real32_t data_d32
    cdef RowData_Real64_t data_d64
    cdef RowData_Complex32_t data_z32
    cdef RowData_Complex64_t data_z64