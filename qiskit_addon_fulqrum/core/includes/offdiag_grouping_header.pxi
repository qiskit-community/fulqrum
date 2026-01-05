# This code is a Qiskit project.
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
from libcpp.vector cimport vector
include "base_header.pxi"


cdef extern from "../src/offdiag_grouping.hpp":

    size_t term_offdiag_structure(const OperatorTerm_t& term) nogil

    void term_offdiag_sort(vector[OperatorTerm_t]& terms) nogil

    unsigned int _max_offdiag_group_size(size_t * ptrs, size_t num_elems) nogil

    void term_group_sort(vector[OperatorTerm_t]& terms, size_t * weight_ptrs, size_t len_ptrs, 
                                unsigned int max_group_size) nogil
