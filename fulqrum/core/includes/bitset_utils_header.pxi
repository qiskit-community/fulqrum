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
from ..core.constants cimport width_t

include "base_header.pxi"


cdef extern from "../src/bitset_utils.hpp":

    void flip_bits(bitset_t& b, const width_t * arr, const width_t size) nogil

    void get_column_bitset(bitset_t& col,
                           const vector[width_t]& pos,
                           const vector[unsigned char]& val,
                           const width_t N) nogil

    width_t bitset_ladder_int(const uint8_t * row,
                                   const width_t * inds,
                                   const width_t ladder_width) nogil

    void compute_orbital_occupancies(const BitsetHashMapWrapper& subspace,
                    const size_t subspace_dim,
                    const double* probabilities,
                    double* out) nogil

    bool passes_proj_validation(const OperatorTerm_t * term,
                                const bitset_t& row) nogil

    vector[width_t] set_bit_indices(bitset_t& row)
