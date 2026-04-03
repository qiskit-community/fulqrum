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
from libcpp.vector cimport vector

include "base_header.pxi"

cdef extern from "../src/operators.hpp":

    void set_extended_flag(OperatorTerm_t& term) nogil

    void set_offdiag_weight_and_phase(OperatorTerm_t& term) nogil

    void combine_qubit_terms(vector[OperatorTerm_t]& terms,
                             vector[OperatorTerm_t]& out_terms,
                             unsigned int * touched,
                             double atol) nogil

    unsigned int term_ladder_int(const OperatorTerm_t& term, unsigned int num_bits) nogil

    void offdiag_weight_sort(QubitOperator_t& oper) nogil

    void set_offdiag_weight_ptrs(vector[OperatorTerm_t]& terms, vector[size_t]& vec) nogil

    unsigned int max_offdiag_ptr_size(size_t * vec, size_t size) nogil
