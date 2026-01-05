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
from fulqrum.core.bitset cimport bitset_t
from fulqrum.core.bitset_hashmap cimport BitsetHashMapWrapper

include "../../core/includes/base_header.pxi"


cdef extern from "../src/simple.hpp":

    double simple_refinement(const OperatorTerm_t * terms,
                             const BitsetHashMapWrapper &subspace,
                             BitsetHashMapWrapper& out_subspace,
                             const vector[OperatorTerm_t]& diag_terms,
                             const unsigned int width,
                             const size_t subspace_dim,
                             const int has_nonzero_diag,
                             const size_t * group_ptrs,
                             const size_t * group_ladder_ptrs,
                             unsigned int * group_rowint_length,
                             vector[vector[unsigned int]]& group_offdiag_inds,
                             const size_t num_groups,
                             const unsigned int ladder_offset,
                             unsigned int max_recursion,
                             double tol
                               ) nogil
