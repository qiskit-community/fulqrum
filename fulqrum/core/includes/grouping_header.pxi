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

cdef extern from "../src/grouping.hpp":


    void compute_term_offdiag_inds(const OperatorTerm_t& term,
                                   unsigned int * offdiag_inds,
                                   unsigned int num_inds) nogil

    void sort_groups_by_ladder_int(QubitOperator_t& oper,
                                 size_t * group_ptrs,
                                 unsigned int num_groups,
                                 unsigned int ladder_width) nogil

    void ladder_bin_starts(const vector[OperatorTerm_t]& terms, const size_t * group_ptrs,
                            unsigned int * group_counts, size_t * group_ranges,
                            unsigned int num_groups, unsigned int num_bins, unsigned int ladder_width) nogil

    void set_group_offdiag_indices(const vector[OperatorTerm_t]& terms,
                                 vector[vector[unsigned int]]& group_indices,
                                 const size_t * group_ptrs,
                                 unsigned int num_groups) nogil
