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
from libc.stdint cimport uint8_t
from libcpp cimport bool
from ..core.bitset cimport bitset_t

include "base_header.pxi"


cdef extern from "../src/bitset_utils.hpp":

    void flip_bits(bitset_t& b, const unsigned int * arr, const unsigned int size) nogil

    void get_column_bitset(bitset_t& col,
                           const unsigned int * pos,
                           const unsigned char * val,
                           const unsigned int N) nogil

    unsigned int bitset_ladder_int(const uint8_t * row, 
                                   const unsigned int * inds,
                                   const unsigned int ladder_width) nogil
    
    void compute_orbital_occupancies(const BitsetHashMapWrapper& subspace,
                    const size_t subspace_dim,
                    const double* probabilities,
                    double* out) nogil
    
    bool passes_proj_validation(const OperatorTerm_t * term,
                                const bitset_t& row) nogil
