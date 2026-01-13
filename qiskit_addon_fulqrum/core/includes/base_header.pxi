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
from ..core.bitset cimport bitset_t
from ..core.bitset_hashmap cimport BitsetHashMapWrapper


cdef extern from "../src/base.hpp":
    ctypedef struct OperatorTerm_t:
        double complex coeff
        vector[unsigned int] indices
        vector[unsigned char] values
        vector[unsigned int] proj_indices
        vector[unsigned int] proj_bits
        unsigned int offdiag_weight
        int extended
        int real_phase
        int group


    ctypedef struct QubitOperator_t:
        unsigned int width
        vector[OperatorTerm_t] terms
        int sorted
        int type
        unsigned int ladder_width
        int weight_sorted
        int off_weight_sorted
        int ladder_sorted


    ctypedef struct Subspace_t:
        BitsetHashMapWrapper bitstrings
        unsigned int num_qubits
        size_t size


    ctypedef struct FermionicTerm_t:
        double complex coeff
        vector[unsigned int] indices
        vector[unsigned char] values

    
    ctypedef struct FermionicOperator_t:
        unsigned int width
        vector[FermionicTerm_t] terms
