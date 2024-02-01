# Fulqrum - Top Hat
# Copyright (C) 2024, IBM
# cython: c_string_type=unicode, c_string_encoding=UTF-8

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.unordered_map cimport unordered_map

from fulqrum_tophat.core.base cimport OperatorTerm

cdef unordered_map[string, size_t] STR_TO_IND = {'Z': 0, '0': 1, '1': 2, 'X': 3, 'Y': 4,
                                                 'D': 5, 'U': 6}

cdef unordered_map[size_t, string] IND_TO_STR = {0: 'Z', 1: '0', 2: '1', 3: 'X', 4: 'Y',
                                                 5: 'D', 6: 'U'}


cdef double complex[28] OPER_ELEMS = [1, 0, 0, -1,    # Z
                                      1, 0, 0, 0,     # 0
                                      0, 0, 0, 1,     # 1
                                      0, 1, 1, 0,     # X
                                      0, -1j, 1j, 0,  # Y
                                      0, 1, 0, 0,     # D
                                      0, 0, 1, 0,     # U
                                     ]

cdef class QubitOperator:
    cdef public size_t num_qubits
    cdef public size_t width
    cdef vector[OperatorTerm] terms
    cdef public bool sorted

    cpdef void append(self, QubitOperator other)
    cpdef bool is_diagonal(self)
