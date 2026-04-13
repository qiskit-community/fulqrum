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
# cython: c_string_type=unicode, c_string_encoding=UTF-8

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.pair cimport pair
from libcpp cimport bool
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
        void sort_term_data()
        void set_proj_indices()
        OperatorTerm_t()
        OperatorTerm_t(complex) except +


    ctypedef struct QubitOperator_t:
        unsigned int width
        vector[OperatorTerm_t] terms
        int sorted
        int type
        unsigned int ladder_width
        int weight_sorted
        int off_weight_sorted
        int ladder_sorted
        QubitOperator_t()
        QubitOperator_t(unsigned int)
        size_t size()
        bool is_real()
        bool is_diagonal()
        QubitOperator_t copy()
        QubitOperator_t& weight_sort()
        QubitOperator_t& offdiag_weight_sort()
        QubitOperator_t& group_sort()
        vector[int] groups()
        vector[size_t] group_ptrs()
        QubitOperator_t& group_term_sort_by_ladder_int(unsigned int)
        QubitOperator_t combine_repeated_terms(double)
        vector[size_t] offdiag_weight_ptrs()
        QubitOperator_t& from_label(string)
        double constant_energy()
        QubitOperator_t remove_constant_terms()
        pair[QubitOperator_t, QubitOperator_t] split_diagonal()
        QubitOperator_t terms_by_group(int)
        vector[int] real_phases()
        vector[complex] coefficients()
        vector[int] extended_terms()
        vector[unsigned int] ladder_integers()
        vector[unsigned int] group_ladder_int_bit_lengths()
        vector[size_t] group_ladder_int_ptrs()


    ctypedef struct Subspace_t:
        BitsetHashMapWrapper bitstrings
        unsigned int num_qubits
        size_t size


    ctypedef struct FermionicTerm_t:
        double complex coeff
        vector[unsigned int] indices
        vector[unsigned char] values
        void insertion_sort()


    ctypedef struct FermionicOperator_t:
        unsigned int width
        vector[FermionicTerm_t] terms
        size_t size()

    size_t max_offdiag_ptr_size(vector[size_t]&)

    OperatorTerm_t& set_proj_indices(OperatorTerm_t&)

    void set_group_offdiag_indices(vector[OperatorTerm_t]& terms,
                                   vector[vector[unsigned int]]& group_indices,
                                   size_t* group_ptrs,
                                   unsigned int num_groups)

    # cdef cppclass TermData:
    #    TermData()
