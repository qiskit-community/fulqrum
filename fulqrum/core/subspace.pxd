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
from .bitset cimport bitset_t

include "includes/base_header.pxi"


cdef class Subspace():
    cdef Subspace_t subspace
    cdef vector[bitset_t] alpha_dets   # populated in half-string mode, empty otherwise
    cdef vector[bitset_t] beta_dets    # populated in half-string mode, empty otherwise
    cdef bint is_half_str              # True if constructed from half-strings
