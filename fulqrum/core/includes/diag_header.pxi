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
from libcpp.pair cimport pair
from ..core.qubit_operator cimport QubitOperator_t
from ..core.bitset cimport bitset_t
from ..core.bitset_hashmap cimport BitsetHashMapWrapper

include "base_header.pxi"

cdef extern from "../src/diag.hpp":
    void compute_diag_vector[T](const BitsetHashMapWrapper& data,
                                T * diag_vec,
                                const QubitOperator_t& diag_oper,
                                const size_t subspace_dim) nogil


    bool fast_diag_compatible(QubitOperator_t&) except + nogil

    void fast_diag_term_sort(QubitOperator_t& oper) except + nogil

    void compute_diag_vector_fast[T](const BitsetHashMapWrapper& data,
                                     T * diag_vec,
                                     const QubitOperator_t& diag_oper,
                                     const size_t subspace_dim) nogil
