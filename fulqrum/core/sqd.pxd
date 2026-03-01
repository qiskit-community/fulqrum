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
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libc.stdint cimport uint32_t, uint64_t
from .bitset cimport bitset_t

cdef extern from "./src/sqd.hpp":

    pair[vector[bitset_t], vector[double]] postselect_bitstrings_cpp(
        const vector[bitset_t] &bitstrings,
        const vector[double] &weights,
        const uint32_t &right,
        const uint32_t &left) nogil

    vector[bitset_t] subsample_cpp(
        const vector[bitset_t] &bitstrings,
        const vector[double] &weights,
        const unsigned int &samples_per_batch,
        const uint32_t &seed) nogil

    pair[vector[bitset_t], vector[double]] recover_configurations_cpp(
        const vector[bitset_t] &bitstrings,
        const vector[double] &weights,
        const vector[double] &avg_occupancies_a,
        const vector[double] &avg_occupancies_b,
        const uint64_t num_elec_a,
        const uint64_t num_elec_b,
        const uint32_t seed) nogil
