# Fulqrum
# Copyright (C) 2024, IBM
# cython: c_string_type=unicode, c_string_encoding=UTF-8

include "includes/base_header.pxi"

cdef class QubitOperator:
    cdef QubitOperator_t oper

    cpdef void append(self, QubitOperator other)
    cpdef int is_diagonal(self)
    cpdef double complex sum_identity_terms(self)