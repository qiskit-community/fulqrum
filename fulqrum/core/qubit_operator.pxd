# Fulqrum - Top Hat
# Copyright (C) 2024, IBM
# cython: c_string_type=unicode, c_string_encoding=UTF-8

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map

from fulqrum.core.base cimport OperatorTerm

cdef unordered_map[char, char] STR_TO_IND = {90: 0, 48: 1, 49: 2, 88: 3,
                                             89: 4, 45: 5, 43: 6}

cdef unordered_map[char, string] IND_TO_STR = {0: 'Z', 1: '0', 2: '1', 3: 'X', 
                                               4: 'Y',5: '-', 6: '+'}


cdef double complex[28] OPER_ELEMS = [1, 0, 0, -1,    # Z
                                      1, 0, 0, 0,     # 0
                                      0, 0, 0, 1,     # 1
                                      0, 1, 1, 0,     # X
                                      0, -1j, 1j, 0,  # Y
                                      0, 1, 0, 0,     # D
                                      0, 0, 1, 0,     # U
                                     ]

cdef class QubitOperator:
    cdef public size_t width
    cdef vector[OperatorTerm] terms
    cdef public bool sorted

    cpdef void append(self, QubitOperator other)
    cpdef bool is_diagonal(self)
    cpdef double complex sum_identity_terms(self)