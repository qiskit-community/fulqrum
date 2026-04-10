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
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libc.string cimport memcpy

import itertools
import math
cimport cython
import numpy as np
cimport numpy as np

from ..exceptions import FulqrumError
from .qubit_operator cimport QubitOperator

include "includes/base_header.pxi"
include "includes/diag_header.pxi"

def proj_index_sort(QubitOperator op):
    diag_proj_index_sort(op.oper)
    return op

def proj_ptrs_and_offset(QubitOperator op):
    cdef pair[vector[size_t], size_t] out = projector_ptrs_and_offset(op.oper)
    cdef size_t[::1] ptrs = np.empty(out.first.size(), np.uintp)
    if out.first.size():
        memcpy(&ptrs[0], &out.first[0], out.first.size()*sizeof(size_t))
    return np.asarray(ptrs), out.second