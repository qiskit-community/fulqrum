# Fulqrum
# Copyright (C) 2024, IBM

from libcpp.vector cimport vector
from fulqrum.core.bitset cimport bitset_t
from fulqrum.core.bitset_hashmap cimport BitsetHashMapWrapper


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
