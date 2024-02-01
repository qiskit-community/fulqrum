# Fulqrum - Top Hat
# Copyright (C) 2024, IBM
# cython: c_string_type=unicode, c_string_encoding=UTF-8
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.unordered_map cimport unordered_map

from fulqrum_tophat.core.base cimport OperatorTerm

cdef unordered_map[string, size_t] STR_TO_IND = {'-': 0, '+': 1, '0': 2, '1': 3}

cdef unordered_map[size_t, string] IND_TO_STR = {0: '-', 1: '+', 2: '0', 3: '1'}


cdef class FermionicOperator:
    cdef public size_t num_orbitals
    cdef public size_t width
    cdef vector[OperatorTerm] terms
    cdef public bool sorted

    cpdef void append(self, FermionicOperator other)
