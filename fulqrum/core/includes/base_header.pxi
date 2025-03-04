# Fulqrum
# Copyright (C) 2024, IBM

from libcpp.vector cimport vector


cdef extern from "../src/base.hpp":
    ctypedef struct OperatorTerm_t:
        double complex coeff
        vector[size_t] indices
        vector[unsigned char] values
        size_t offdiag_weight
        int extended
        int group


    ctypedef struct QubitOperator_t:
        size_t width
        vector[OperatorTerm_t] terms
        int sorted 


    ctypedef struct Subspace_t:
        vector[unsigned char] bitstrings
        vector[size_t] bin_counts
        vector[size_t] bin_ranges
        size_t num_qubits
        size_t num_bins
        size_t bin_width
        size_t size
